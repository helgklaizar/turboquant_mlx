import AVFoundation
import Foundation

enum AudioCaptureError: LocalizedError {
    case microphoneDenied
    case engineFailed(String)

    var errorDescription: String? {
        switch self {
        case .microphoneDenied:
            return "Microphone access was denied. Enable it in Settings › TurboMic."
        case .engineFailed(let detail):
            return "Audio engine failed: \(detail)"
        }
    }
}

/// Owns `AVAudioSession` + `AVAudioEngine`.
///
/// Publishes raw PCM buffers to whoever is listening (the transcriber) and, when
/// asked, mirrors the same buffers into an `.caf` file so the raw audio survives
/// the session.
final class AudioCaptureService {

    private let engine = AVAudioEngine()
    private var audioFile: AVAudioFile?
    private var isRunning = false

    /// Called on the audio thread — do not touch UI state directly from here.
    var onBuffer: ((AVAudioPCMBuffer) -> Void)?
    /// Rough input level in 0...1, useful for the button animation.
    var onLevel: ((Float) -> Void)?

    static func requestPermission() async -> Bool {
        await withCheckedContinuation { continuation in
            AVAudioApplication.requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
    }

    /// Configures the session so capture keeps running when the app is backgrounded
    /// and other audio (music, a call ringtone) is not killed.
    private func configureSession() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(
            .playAndRecord,
            mode: .spokenAudio,
            options: [.mixWithOthers, .allowBluetooth, .defaultToSpeaker]
        )
        try session.setActive(true, options: .notifyOthersOnDeactivation)
    }

    func start(recordingTo fileURL: URL?) throws {
        guard !isRunning else { return }
        try configureSession()

        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)

        if let fileURL {
            do {
                audioFile = try AVAudioFile(forWriting: fileURL, settings: format.settings)
            } catch {
                // Losing the archive copy is not a reason to lose the session.
                audioFile = nil
            }
        }

        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, _ in
            guard let self else { return }
            self.onBuffer?(buffer)
            try? self.audioFile?.write(from: buffer)
            self.onLevel?(Self.level(of: buffer))
        }

        engine.prepare()
        do {
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            audioFile = nil
            throw AudioCaptureError.engineFailed(error.localizedDescription)
        }
        isRunning = true
    }

    func stop() {
        guard isRunning else { return }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        audioFile = nil
        isRunning = false
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    private static func level(of buffer: AVAudioPCMBuffer) -> Float {
        guard let channel = buffer.floatChannelData?[0] else { return 0 }
        let count = Int(buffer.frameLength)
        guard count > 0 else { return 0 }

        var sum: Float = 0
        for index in 0..<count {
            let sample = channel[index]
            sum += sample * sample
        }
        let rms = (sum / Float(count)).squareRoot()
        // Map roughly -50 dBFS ... 0 dBFS onto 0...1.
        let db = 20 * log10(max(rms, 1e-7))
        return min(max((db + 50) / 50, 0), 1)
    }
}
