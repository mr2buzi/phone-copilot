package com.phonecopilot.helper

import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import java.time.Instant
import java.util.concurrent.Executors
import org.json.JSONObject

class PhoneCopilotNotificationService : NotificationListenerService() {
    private val executor = Executors.newSingleThreadExecutor()

    override fun onNotificationPosted(sbn: StatusBarNotification) {
        val extras = sbn.notification.extras
        val payload = JSONObject()
            .put("package_name", sbn.packageName)
            .put("title", extras.getString("android.title"))
            .put("text", extras.getCharSequence("android.text")?.toString())
            .put("posted_at", Instant.ofEpochMilli(sbn.postTime).toString())
        executor.execute {
            forwardPayload(payload)
        }
    }

    private fun forwardPayload(payload: JSONObject) {
        val connection = URL(CONTROLLER_ENDPOINT).openConnection() as HttpURLConnection
        try {
            connection.requestMethod = "POST"
            connection.connectTimeout = 3000
            connection.readTimeout = 3000
            connection.doOutput = true
            connection.setRequestProperty("Content-Type", "application/json")
            OutputStreamWriter(connection.outputStream).use { writer ->
                writer.write(payload.toString())
            }
            connection.inputStream.close()
        } catch (_: Exception) {
            // The helper stays quiet when the desktop controller is offline.
        } finally {
            connection.disconnect()
        }
    }

    companion object {
        private const val CONTROLLER_ENDPOINT = "http://127.0.0.1:8765/api/notifications"
    }
}
