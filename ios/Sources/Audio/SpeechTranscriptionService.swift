import AVFoundation
import Foundation
import Speech

enum TranscriptionError: LocalizedError {
    case notAuthorized
    case recognizerUnavailable(String)

    var errorDescription: String? {
        switch self {
        case .notAuthorized:
            return "Speech recognition was denied. Enable it in Settings › TurboMic."
        case .recognizerUnavailable(let locale):
            return "On-device speech recognition is unavailable for \(locale). Download the language in iOS Settings › General › Keyboard › Dictation."
        }
    }
}

/// Streaming speech-to-text on top of `SFSpeechRecognizer`.
///
/// Two things make this more than a textbook wrapper:
///
/// 1. **Rotation.** A single `SFSpeechRecognitionTask` does not run forever — iOS
///    ends it after roughly a minute. For an assistant that is supposed to listen
///    for an hour we rotate the request on a timer, commit whatever was final, and
///    open a fresh one without dropping audio.
/// 2. **On-device.** `requiresOnDeviceRecognition` keeps audio off Apple's servers,
///    which is the whole point of pairing this with a local model.
final class SpeechTranscriptionService {

    /// Fires for every partial hypothesis of the segment currently in flight.
    var onPartial: ((String) -> Void)?
    /// Fires once a segment is committed and will not change any more.
    var onFinalSegment: ((String) -> Void)?
    var onError: ((Error) -> Void)?

    private let recognizer: SFSpeechRecognizer?
    private let localeIdentifier: String
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var rotationTimer: Timer?
    private var lastPartial: String = ""
    private var isRunning = false

    /// A single task is cut well before the system would cut it for us.
    private let rotationInterval: TimeInterval = 50

    init(localeIdentifier: String) {
        self.localeIdentifier = localeIdentifier
        self.recognizer = SFSpeechRecognizer(locale: Locale(identifier: localeIdentifier))
    }

    static func requestPermission() async -> Bool {
        await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status == .authorized)
            }
        }
    }

    func start() throws {
        guard SFSpeechRecognizer.authorizationStatus() == .authorized else {
            throw TranscriptionError.notAuthorized
        }
        guard let recognizer, recognizer.isAvailable else {
            throw TranscriptionError.recognizerUnavailable(localeIdentifier)
        }
        isRunning = true
        startSegment()
        scheduleRotation()
    }

    func stop() {
        isRunning = false
        rotationTimer?.invalidate()
        rotationTimer = nil
        commitPartial()
        request?.endAudio()
        task?.cancel()
        request = nil
        task = nil
    }

    /// Feed audio in from `AudioCaptureService`.
    func append(_ buffer: AVAudioPCMBuffer) {
        request?.append(buffer)
    }

    // MARK: - Segments

    private func startSegment() {
        guard let recognizer else { return }

        let newRequest = SFSpeechAudioBufferRecognitionRequest()
        newRequest.shouldReportPartialResults = true
        newRequest.requiresOnDeviceRecognition = true
        if #available(iOS 16.0, *) {
            newRequest.addsPunctuation = true
        }
        request = newRequest
        lastPartial = ""

        task = recognizer.recognitionTask(with: newRequest) { [weak self] result, error in
            guard let self else { return }

            if let result {
                let text = result.bestTranscription.formattedString
                self.lastPartial = text
                self.onPartial?(text)
                if result.isFinal {
                    self.commitPartial()
                }
            }

            if let error {
                // A cancelled task during rotation or stop is expected, not a failure.
                let nsError = error as NSError
                let isCancellation = nsError.code == 203 || nsError.code == 216 || nsError.code == 301
                if self.isRunning && !isCancellation {
                    self.onError?(error)
                }
            }
        }
    }

    private func scheduleRotation() {
        rotationTimer?.invalidate()
        rotationTimer = Timer.scheduledTimer(withTimeInterval: rotationInterval, repeats: true) { [weak self] _ in
            self?.rotate()
        }
    }

    /// Close the current recognition task and immediately open the next one.
    private func rotate() {
        guard isRunning else { return }
        commitPartial()
        request?.endAudio()
        task?.cancel()
        request = nil
        task = nil
        startSegment()
    }

    private func commitPartial() {
        let text = lastPartial.trimmingCharacters(in: .whitespacesAndNewlines)
        lastPartial = ""
        guard !text.isEmpty else { return }
        onFinalSegment?(text)
    }
}
