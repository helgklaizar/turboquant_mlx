import SwiftUI

@main
struct TurboMicApp: App {
    @StateObject private var settings = AppSettings()
    @StateObject private var store = SessionStore()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(settings)
                .environmentObject(store)
        }
    }
}
