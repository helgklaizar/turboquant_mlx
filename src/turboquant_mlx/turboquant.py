import mlx.core as mx
from core.qjl import QJLCompressor
from core.polarquant import PolarQuantCompressor

class TurboQuant:
    def __init__(self, feature_dim: int, pq_bits: int = 3, qjl_features: int = 2048, seed: int = 42):
        """
        TurboQuant Compressor (Two-stage pipeline for KV-Cache) natively using MLX.
        
        1. Uses PolarQuant (MSE-optimal quantization).
        2. Computes the error (residual).
        3. Quantizes the error via QJL to remove dot-product bias.
        
        :param feature_dim: vector dimensionality (d), must be a power of two
        :param pq_bits: bit rate for angle quantization in PolarQuant
        :param qjl_features: number of random features (k) for QJL 
        """
        self.feature_dim = feature_dim
        
        # Base quantizer (minimizes L2 distance)
        self.pq = PolarQuantCompressor(feature_dim=feature_dim, theta_bits=pq_bits, seed=seed)
        
        # 1-bit dot-product corrector (residual)
        self.qjl = QJLCompressor(feature_dim=feature_dim, num_features=qjl_features, seed=seed+1)
        
    def compress(self, x: mx.array) -> dict:
        """
        Two-stage vector or batch compression natively using MLX.
        :param x: vector (d,) or batch (b, d)
        :return: dictionary with compressed data
        """
        # --- STAGE 1: MSE Quantization (PolarQuant) ---
        pq_compressed = self.pq.compress(x)
        
        # Reconstruct the approximated vector
        x_mse_approx = self.pq.decompress(pq_compressed)
        
        # --- STAGE 2: Residual + QJL ---
        residual = x - x_mse_approx
        
        # Quantize the residual via QJL to 1 bit (sign) + store L2 norm (one number per vector)
        qjl_quant, qjl_norm = self.qjl.compress(residual)
        
        return {
            "pq_data": pq_compressed,
            "qjl_data": qjl_quant,
            "qjl_norm": qjl_norm
        }
        
    def estimate_dot(self, compressed: dict, y: mx.array) -> mx.array:
        """
        Unbiased estimation of the dot product using MLX.
        :param compressed: compressed data from compress()
        :param y: original uncompressed query (float-vector of shape (d,))
        :return: estimation x * y
        """
        # 1. Classical dot product of the approximated vector
        x_mse_approx = self.pq.decompress(compressed["pq_data"])
        
        # Handle shapes: batch/single x queries
        if x_mse_approx.ndim == 2 and y.ndim == 1:
            dot_mse = x_mse_approx @ y
        else:
            dot_mse = x_mse_approx @ y.T
            
        # 2. Estimation of residual dot product (compensates dot_mse bias)
        dot_residual = self.qjl.estimate_dot(
            x_quant=compressed["qjl_data"], 
            norm_x=compressed["qjl_norm"], 
            y=y
        )
        
        # Final dot product = PolarQuant share + Corrected error share
        return dot_mse + dot_residual
