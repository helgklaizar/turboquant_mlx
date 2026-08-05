import SwiftUI

struct ContentView: View {

    @EnvironmentObject private var settings: AppSettings
    @EnvironmentObject private var store: SessionStore
    @StateObject private var recorder = RecorderViewModel()
    @State private var showSettings = false
    @State private var showSessions = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                modePicker
                    .padding(.horizontal)
                    .disabled(recorder.state.isActive)

                Spacer(minLength: 0)

                RecordButton(state: recorder.state, level: recorder.level) {
                    Task { await recorder.toggle() }
                }

                statusLine

                Spacer(minLength: 0)

                liveOutput
            }
            .padding(.vertical)
            .navigationTitle("TurboMic")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button { showSessions = true } label: {
                        Image(systemName: "list.bullet.rectangle")
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button { showSettings = true } label: {
                        Image(systemName: "gearshape")
                    }
                }
            }
            .sheet(isPresented: $showSettings) { SettingsView() }
            .sheet(isPresented: $showSessions) { SessionListView() }
        }
        .onAppear { recorder.bind(settings: settings, store: store) }
    }

    private var modePicker: some View {
        VStack(spacing: 6) {
            Picker("Mode", selection: Binding(
                get: { settings.mode },
                set: { settings.mode = $0 }
            )) {
                ForEach(CaptureMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            Text(settings.mode.subtitle)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var statusLine: some View {
        VStack(spacing: 4) {
            switch recorder.state {
            case .listening:
                Text(Self.clock(recorder.elapsed))
                    .font(.system(.title3, design: .monospaced))
                    .foregroundStyle(.secondary)
            case .failed(let message):
                Text(message)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            default:
                Text(" ").font(.title3)
            }

            if recorder.pipeline.isAnalyzing {
                Label("Analysing…", systemImage: "sparkles")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if let error = recorder.pipeline.lastError {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
            }
        }
    }

    @ViewBuilder
    private var liveOutput: some View {
        if recorder.pipeline.insights.isEmpty && recorder.liveText.isEmpty {
            Text("Tap the button and talk. Only what matters is kept.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .padding(.bottom, 24)
        } else {
            TabView {
                InsightListView(
                    summary: recorder.pipeline.summary,
                    insights: recorder.pipeline.insights
                )
                .tabItem { Label("Insights", systemImage: "sparkles") }

                TranscriptView(
                    committed: recorder.transcript,
                    partial: recorder.partial
                )
                .tabItem { Label("Transcript", systemImage: "text.alignleft") }
            }
            .frame(height: 300)
        }
    }

    private static func clock(_ interval: TimeInterval) -> String {
        let total = Int(interval)
        return String(format: "%02d:%02d", total / 60, total % 60)
    }
}

struct TranscriptView: View {
    let committed: String
    let partial: String

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 8) {
                Text(committed)
                    .textSelection(.enabled)
                if !partial.isEmpty {
                    Text(partial)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding()
        }
    }
}
