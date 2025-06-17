import paho.mqtt.client as mqtt
import json
import requests
 
# ThingsBoard MQTT Settings
THINGSBOARD_HOST = "mqtt.thingsboard.cloud"
THINGSBOARD_PORT = 1883
THINGSBOARD_ACCESS_TOKEN = "g61tzxgs0u0u8xf5yjyx"
THINGSBOARD_TOPIC = "v1/devices/me/telemetry"
 
# ThingsBoard HTTP API endpoint
THINGSBOARD_HTTP_URL = f"http://thingsboard.cloud/api/v1/{THINGSBOARD_ACCESS_TOKEN}/telemetry"
 
# Create MQTT client
client = mqtt.Client(client_id="PlatePublisher")
 
def connect_mqtt():
    try:
        client.username_pw_set(THINGSBOARD_ACCESS_TOKEN)
        client.connect(THINGSBOARD_HOST, THINGSBOARD_PORT, keepalive=60)
        client.loop_start()
        print("[MQTT] Connected to ThingsBoard.")
    except Exception as e:
        print(f"[MQTT] MQTT connection failed: {e}")
 
def publish_plate(plate, confidence, timestamp):
    message = {
        "plate": plate,
        "confidence": round(confidence, 2),
        "timestamp": timestamp
    }
 
    # Send via MQTT
    try:
        client.publish(THINGSBOARD_TOPIC, json.dumps(message), qos=2)
        print(f"[MQTT] Published: {message}")
    except Exception as e:
        print(f"[MQTT] Error: {e}")
 
    # Send via HTTP POST
    try:
        response = requests.post(
            THINGSBOARD_HTTP_URL,
            headers={"Content-Type": "application/json"},
            json=message
        )
        if response.status_code == 200:
            #print(f"[HTTP] Successfully posted: {message}")
            pass
        else:
            print(f"[HTTP] HTTP {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[HTTP] Error: {e}")