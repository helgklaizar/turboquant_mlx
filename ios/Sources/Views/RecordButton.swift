import SwiftUI

/// The one control the app is built around: a large circle in the middle of the
/// screen. It pulses with the input level while listening so it is obvious, at a
/// glance across a room, that the mic is live.
struct RecordButton: View {

    let state: RecorderState
    let level: Float
    let action: () -> Void

    private var tint: Color {
        switch state {
        case .idle: return .accentColor
        case .preparing: return .orange
        case .listening: return .red
        case .finishing: return .purple
        case .failed: return .gray
        }
    }

    private var label: String {
        switch state {
        case .idle: return "Record"
        case .preparing: return "Starting"
        case .listening: return "Stop"
        case .finishing: return "Analysing"
        case .failed: return "Retry"
        }
    }

    private var icon: String {
        switch state {
        case .idle, .failed: return "mic.fill"
        case .preparing: return "hourglass"
        case .listening: return "stop.fill"
        case .finishing: return "sparkles"
        }
    }

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle()
                    .fill(tint.opacity(0.16))
                    .frame(width: 260, height: 260)
                    .scaleEffect(state == .listening ? 1 + CGFloat(level) * 0.25 : 1)
                    .animation(.easeOut(duration: 0.12), value: level)

                Circle()
                    .fill(tint)
                    .frame(width: 190, height: 190)
                    .shadow(color: tint.opacity(0.45), radius: 24, y: 8)

                VStack(spacing: 10) {
                    Image(systemName: icon)
                        .font(.system(size: 52, weight: .semibold))
                    Text(label)
                        .font(.headline)
                }
                .foregroundStyle(.white)
            }
        }
        .buttonStyle(.plain)
        .disabled(state == .finishing)
        .accessibilityLabel(label)
        .accessibilityHint("Starts or stops recording and analysis")
    }
}
