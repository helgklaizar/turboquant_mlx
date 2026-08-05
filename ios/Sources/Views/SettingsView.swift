import SwiftUI

struct SettingsView: View {
    @EnvironmentObject private var settings: AppSettings
    @Environment(\.dismiss) private var dismiss

    @State private var healthMessage: String?
    @State private var isChecking = false

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("http://mac.local:8080", text: $settings.serverURL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.URL)
                    TextField("Model", text: $settings.modelName)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()

                    Button {
                        Task { await checkHealth() }
                    } label: {
                        HStack {
                            Text("Test connection")
                            if isChecking {
                                Spacer()
                                ProgressView()
                            }
                        }
                    }

                    if let healthMessage {
                        Text(healthMessage)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } header: {
                    Text("Backend")
                } footer: {
                    Text("Your Mac running `python scripts/run_assistant_server.py`. Any OpenAI-compatible endpoint also works.")
                }

                Section("Recognition") {
                    Picker("Language", selection: $settings.localeIdentifier) {
                        ForEach(AppSettings.supportedLocales, id: \.id) { locale in
                            Text(locale.name).tag(locale.id)
                        }
                    }
                }

                Section {
                    Toggle("Keep audio files", isOn: $settings.keepAudio)
                } footer: {
                    Text("Off by default. Transcripts and extracted items are always kept on this device.")
                }

                Section {
                    Text("Speech recognition runs on-device. Transcripts leave the phone only to reach the backend address above.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Settings")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private func checkHealth() async {
        isChecking = true
        defer { isChecking = false }
        do {
            let analyzer = try RemoteAnalyzer(baseURLString: settings.serverURL, model: settings.modelName)
            healthMessage = try await analyzer.checkHealth()
        } catch {
            healthMessage = error.localizedDescription
        }
    }
}
