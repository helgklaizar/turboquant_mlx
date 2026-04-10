import mlx.core as mx

class TurboQuantKVCache:
    """
    KVCache реализация для Apple MLX.
    Заменяет стандартный mlx_lm.models.cache.KVCache на нашу сжатую версию TurboQuant.
    Она сжимает ключи (и значения по желанию) во время префил-фазы генерации.
    Поддерживает lazy initialization — компрессоры создаются при первом вызове update_and_fetch,
    что обеспечивает совместимость с Qwen/GQA-моделями (они вызывают KVCache() без аргументов).
    """
    step = 256  # для совместимости с mlx_lm server

    def __init__(self, head_dim: int = 0, n_kv_heads: int = 0, k_theta_bits: int = 8, k_radius_bits: int = 8, v_theta_bits: int = 3, v_radius_bits: int = 8, fp16_sink_size: int = 128, is_boundary: bool = False):
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        
        self.offset = 0
        self.chunk_size = 64
        
        self.k_theta_bits = k_theta_bits
        self.k_radius_bits = k_radius_bits
        self.v_theta_bits = v_theta_bits
        self.v_radius_bits = v_radius_bits
        self.fp16_sink_size_default = fp16_sink_size
        self.is_boundary = is_boundary
        
        # Отложенная инициализация — компрессор создаётся при первом вызове update_and_fetch
        self._initialized = False
        self.compress_k = False
        self.compress_v = False
        self.fp16_sink_size = float('inf')  # По умолчанию не сжимаем до инициализации
        
        # Буфер для первых важных токенов без сжатия (Attention Sink) или для boundary-слоев
        self.sink_keys = None
        self.sink_values = None
        
        self.compressed_keys_chunks = []
        self.compressed_values_chunks = []
        
        self.uncompressed_keys_chunks = []
        self.uncompressed_values_chunks = []
        
        self.key_buffer = None
        self.value_buffer = None
        
        # Атрибуты для совместимости с mlx_lm server
        self.keys = None
        self.values = None

    def _lazy_init(self, head_dim: int, n_kv_heads: int):
        """Инициализирует компрессоры на основе реальной размерности тензора."""
        if self._initialized:
            return
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        
        if self.is_boundary or (self.k_theta_bits >= 16 and self.v_theta_bits >= 16):
            self.fp16_sink_size = float('inf')
            self.compress_k = False
            self.compress_v = False
        else:
            self.fp16_sink_size = self.fp16_sink_size_default
            self.compress_k = self.k_theta_bits < 16
            self.compress_v = self.v_theta_bits < 16
            
            from core.polarquant import PolarQuantCompressor
            if self.compress_k:
                self.k_compressor = PolarQuantCompressor(
                    feature_dim=head_dim, 
                    theta_bits=self.k_theta_bits, 
                    radius_bits=self.k_radius_bits
                )
            if self.compress_v:
                self.v_compressor = PolarQuantCompressor(
                    feature_dim=head_dim, 
                    theta_bits=self.v_theta_bits, 
                    radius_bits=self.v_radius_bits
                )
        
        self._initialized = True

    def _compress_and_store(self, k: mx.array, v: mx.array):
        b, h, s, d = k.shape
        
        if self.compress_k:
            k_2d = mx.reshape(k, (-1, d))
            compressed_k = self.k_compressor.compress(k_2d)
            self.compressed_keys_chunks.append((compressed_k, (b, h, s, d)))
        else:
            self.uncompressed_keys_chunks.append(k)
            
        if self.compress_v:
            v_2d = mx.reshape(v, (-1, d))
            compressed_v = self.v_compressor.compress(v_2d)
            self.compressed_values_chunks.append((compressed_v, (b, h, s, d)))
        else:
            self.uncompressed_values_chunks.append(v)

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        # Lazy init: определяем размерности из реального тензора
        b, h, s, d = keys.shape
        self._lazy_init(head_dim=d, n_kv_heads=h)
        
        prev_offset = self.offset
        self.offset += keys.shape[2]
        
        # 1. Логика Attention Sink (или Boundary Layers)
        if prev_offset < self.fp16_sink_size:
            remaining_sink = self.fp16_sink_size - prev_offset
            
            # Забираем токены, которые влезают в Sink
            k_sink_part = keys[:, :, :remaining_sink, :]
            v_sink_part = values[:, :, :remaining_sink, :]
            
            if self.sink_keys is None:
                self.sink_keys = k_sink_part
                self.sink_values = v_sink_part
            else:
                self.sink_keys = mx.concatenate([self.sink_keys, k_sink_part], axis=2)
                self.sink_values = mx.concatenate([self.sink_values, v_sink_part], axis=2)
                
            k_compress_part = keys[:, :, remaining_sink:, :]
            v_compress_part = values[:, :, remaining_sink:, :]
        else:
            k_compress_part = keys
            v_compress_part = values
            
        # 2. Логика сжатия для оставшихся
        if k_compress_part.shape[2] > 0:
            if self.key_buffer is None:
                self.key_buffer = k_compress_part
                self.value_buffer = v_compress_part
            else:
                self.key_buffer = mx.concatenate([self.key_buffer, k_compress_part], axis=2)
                self.value_buffer = mx.concatenate([self.value_buffer, v_compress_part], axis=2)
                
            while self.key_buffer is not None and self.key_buffer.shape[2] >= self.chunk_size:
                chunk_k = self.key_buffer[:, :, :self.chunk_size, :]
                chunk_v = self.value_buffer[:, :, :self.chunk_size, :]
                
                self._compress_and_store(chunk_k, chunk_v)
                
                if self.key_buffer.shape[2] > self.chunk_size:
                    self.key_buffer = self.key_buffer[:, :, self.chunk_size:, :]
                    self.value_buffer = self.value_buffer[:, :, self.chunk_size:, :]
                else:
                    self.key_buffer = None
                    self.value_buffer = None
                
        # 3. Декомпрессия старых чанков в единый кэш
        full_keys = []
        full_values = []
        
        if self.sink_keys is not None:
            full_keys.append(self.sink_keys)
            full_values.append(self.sink_values)
            
        # Восстанавливаем k
        k_chunk_idx = 0
        comp_k_idx = 0
        uncomp_k_idx = 0
        while comp_k_idx < len(self.compressed_keys_chunks) or uncomp_k_idx < len(self.uncompressed_keys_chunks):
            if self.compress_k:
                comp_k, shape = self.compressed_keys_chunks[comp_k_idx]
                k_approx_2d = self.k_compressor.decompress(comp_k)
                full_keys.append(mx.reshape(k_approx_2d, shape))
                comp_k_idx += 1
            else:
                full_keys.append(self.uncompressed_keys_chunks[uncomp_k_idx])
                uncomp_k_idx += 1
                
        # Восстанавливаем v
        comp_v_idx = 0
        uncomp_v_idx = 0
        while comp_v_idx < len(self.compressed_values_chunks) or uncomp_v_idx < len(self.uncompressed_values_chunks):
            if self.compress_v:
                comp_v, shape = self.compressed_values_chunks[comp_v_idx]
                v_approx_2d = self.v_compressor.decompress(comp_v)
                full_values.append(mx.reshape(v_approx_2d, shape))
                comp_v_idx += 1
            else:
                full_values.append(self.uncompressed_values_chunks[uncomp_v_idx])
                uncomp_v_idx += 1
            
        if self.key_buffer is not None:
            full_keys.append(self.key_buffer)
            full_values.append(self.value_buffer)
            
        if not full_keys:
            return keys, values
            
        return mx.concatenate(full_keys, axis=2), mx.concatenate(full_values, axis=2)

    @property
    def state(self):
        k = []
        v = []
        if self.sink_keys is not None:
            k.append(self.sink_keys)
            v.append(self.sink_values)
            
        if self.compress_k:
            for comp_k, shape in self.compressed_keys_chunks:
                k.append(mx.reshape(self.k_compressor.decompress(comp_k), shape))
        else:
            for uncomp_k in self.uncompressed_keys_chunks:
                k.append(uncomp_k)
                
        if self.compress_v:
            for comp_v, shape in self.compressed_values_chunks:
                v.append(mx.reshape(self.v_compressor.decompress(comp_v), shape))
        else:
            for uncomp_v in self.uncompressed_values_chunks:
                v.append(uncomp_v)
                
        if self.key_buffer is not None:
            k.append(self.key_buffer)
            v.append(self.value_buffer)
            
        ret_k = mx.concatenate(k, axis=2) if k else mx.array([])
        ret_v = mx.concatenate(v, axis=2) if v else mx.array([])
        return ret_k, ret_v
        
    @property
    def memory_size(self):
        total_bytes = 0
        for t in [self.sink_keys, self.sink_values, self.key_buffer, self.value_buffer] + self.uncompressed_keys_chunks + self.uncompressed_values_chunks:
            if t is not None:
                total_bytes += t.size * 2
                
        for comp, _ in self.compressed_keys_chunks:
            total_bytes += sum(a.size * a.dtype.size for a in comp["angles"])
            if isinstance(comp.get("q_radius"), mx.array):
                total_bytes += comp["q_radius"].size * comp["q_radius"].dtype.size
                
        for comp, _ in self.compressed_values_chunks:
            total_bytes += sum(a.size * a.dtype.size for a in comp["angles"])
            if isinstance(comp.get("q_radius"), mx.array):
                total_bytes += comp["q_radius"].size * comp["q_radius"].dtype.size
            
        return total_bytes

def apply_turboquant_cache(model=None, k_theta_bits: int = 8, k_radius_bits: int = 8, v_theta_bits: int = 3, v_radius_bits: int = 8, fp16_sink_size: int = 128):
    try:
        import mlx_lm.models.cache as cache_module
    except ImportError:
        print("[TurboQuant] Ошибка: mlx-lm не установлен.")
        return
        
    class PatchedCache(TurboQuantKVCache):
        def __init__(self, head_dim: int, n_kv_heads: int, is_boundary: bool = False, **kwargs):
            super().__init__(
                head_dim=head_dim, 
                n_kv_heads=n_kv_heads, 
                k_theta_bits=k_theta_bits, 
                k_radius_bits=k_radius_bits,
                v_theta_bits=v_theta_bits,
                v_radius_bits=v_radius_bits,
                fp16_sink_size=fp16_sink_size,
                is_boundary=is_boundary
            )

    cache_module.KVCache = PatchedCache
    
    if hasattr(cache_module, 'make_prompt_cache'):
        _original_make = cache_module.make_prompt_cache
        def patched_make_prompt_cache(model, max_kv_size=None):
            if hasattr(model, "make_cache"):
                return model.make_cache()
            
            caches = []
            num_layers = len(model.layers)
            for i, l in enumerate(model.layers):
                is_boundary = i < 2 or i >= num_layers - 2
                
                h_dim = getattr(l, "head_dim", None)
                if h_dim is None and hasattr(l, "self_attn"):
                    h_dim = getattr(l.self_attn, "head_dim", None)
                if h_dim is None and hasattr(model, "args"):
                    args = model.args
                    if hasattr(args, "head_dim"): h_dim = args.head_dim
                    elif hasattr(args, "hidden_size") and hasattr(args, "num_attention_heads"):
                        h_dim = args.hidden_size // args.num_attention_heads
                h_dim = h_dim or 128
                        
                n_kv = getattr(l, "n_kv_heads", None)
                if n_kv is None and hasattr(l, "self_attn"):
                    n_kv = getattr(l.self_attn, "num_key_value_heads", getattr(l.self_attn, "n_kv_heads", None))
                n_kv = n_kv or getattr(l, "n_heads", 8)

                caches.append(PatchedCache(head_dim=h_dim, n_kv_heads=n_kv, is_boundary=is_boundary))
            return caches
        cache_module.make_prompt_cache = patched_make_prompt_cache

    print(f"[TurboQuant] Глобальный патч установлен: mlx_lm.models.cache.KVCache подменен.")
    print(f"[TurboQuant] Настройки: K-bits (theta={k_theta_bits}, radius={k_radius_bits}), V-bits (theta={v_theta_bits}, radius={v_radius_bits}). Attention Sink: первые {fp16_sink_size} токенов.")
    print(f"[TurboQuant] Умная слоистая изоляция (Boundary V) включена для первых и последних 2 слоев.")
