/**
   BasicHTTPSClient.ino

    Created on: 20.08.2018

*/

#include <Arduino.h>

#include <ESP8266WiFi.h>
#include <ESP8266WiFiMulti.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClientSecureBearSSL.h>

#include "certs.h"
#include "IoTvars_TaSciEn.h" // change name of file IoTvars_default.h to IoTvars.h




ESP8266WiFiMulti WiFiMulti;

const int buttonPin = BUTTONPIN;

int old_door_state = 0;
int door_state = 0;
int timer = 0;
void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  //pinMode(8, INPUT);
  Serial.begin(115200);
  // Serial.setDebugOutput(true);

  Serial.println();
  Serial.println();
  Serial.println();

  WiFi.mode(WIFI_STA);
  WiFiMulti.addAP(STASSID, STAPSK);
  Serial.println("setup() done connecting to ssid '" STASSID "'");

  
  old_door_state = digitalRead(buttonPin);
  show_running();
  https_stuff(old_door_state, 0);
}

void loop() {
  int door_state = digitalRead(buttonPin);
  if (door_state != old_door_state){
    Serial.print("[ModeemiDoorbot] Change detected, waiting 15 seconds...\n");
    delay(15000);
    int verify_door_state = digitalRead(buttonPin);
    if (verify_door_state == door_state){
      
      door_state = verify_door_state;
      old_door_state = verify_door_state;
      if(verify_door_state == HIGH){
        Serial.print("[ModeemiDoorbot] New state is CLOSED, Sending update to server...\n");
        digitalWrite(LED_BUILTIN, LOW);
      }else{
        Serial.print("[ModeemiDoorbot] New state is OPEN, Sending update to server...\n");
        digitalWrite(LED_BUILTIN, HIGH);
      }
      https_stuff(verify_door_state, 0);
    }

  }
  //https_stuff(door_state);
  show_armed();
  Serial.println("[ModeemiDoorbot] Waiting 4s...");
  delay(10000);
  timer = timer + 1;
  if(timer == TIMER0){
    timer = 0;
    https_stuff(door_state, 1);
  }
}



void https_stuff(int door_state, int keepalive){
  if ((WiFiMulti.run() == WL_CONNECTED)) {

    std::unique_ptr<BearSSL::WiFiClientSecure> client(new BearSSL::WiFiClientSecure);

    // client->setFingerprint(fingerprint_sni_cloudflaressl_com);
    // Or, if you happy to ignore the SSL certificate, then use the following line instead:
    client->setInsecure();

    HTTPClient https;

    Serial.print("[HTTPS] begin...\n");
    String httpAddress;
    String spaceid = SPACEID;
    if(keepalive == 0){
      if(door_state == INVERT_SET){
        httpAddress = "https://www.doors.modeemi.fi/space_events/"+spaceid+"/open";
      } else {
        httpAddress = "https://www.doors.modeemi.fi/space_events/"+spaceid+"/close";
      }
    }else{
      if(door_state == INVERT_SET){
        httpAddress = "https://www.doors.modeemi.fi/space_events/"+spaceid+"/keepalive/open";
      } else {
        httpAddress = "https://www.doors.modeemi.fi/space_events/"+spaceid+"/keepalive/close";
      }
    }

    
      
    

    if (https.begin(*client, httpAddress)) {  // HTTPS
      https.setAuthorization(WEB_USER, WEB_PASS);
      Serial.print("[HTTPS] POST...\n");
      // start connection and send HTTP header
      int httpCode = https.POST("");

      // httpCode will be negative on error
      if (httpCode > 0) {
        // HTTP header has been send and Server response header has been handled
        Serial.printf("[HTTPS] POST... code: %d\n", httpCode);

        // file found at server
        if (httpCode == HTTP_CODE_OK || httpCode == HTTP_CODE_MOVED_PERMANENTLY) {
          String payload = https.getString();
          Serial.println(payload);
        }
      } else {
        Serial.printf("[HTTPS] GET... failed, error: %s\n", https.errorToString(httpCode).c_str());
      }

      https.end();
    } else {
      Serial.printf("[HTTPS] Unable to connect\n");
    }
  }
}



void show_running(){
  digitalWrite(LED_BUILTIN, LOW);
  delay(100);
  digitalWrite(LED_BUILTIN, HIGH);
  delay(600);

  digitalWrite(LED_BUILTIN, LOW);
  delay(100);
  digitalWrite(LED_BUILTIN, HIGH);
  delay(100);
  digitalWrite(LED_BUILTIN, LOW);
  delay(100);
  digitalWrite(LED_BUILTIN, HIGH);
}

void show_armed(){
  digitalWrite(LED_BUILTIN, LOW);
  delay(500);
  digitalWrite(LED_BUILTIN, HIGH);
  delay(500);

  digitalWrite(LED_BUILTIN, LOW);
  delay(100);
  digitalWrite(LED_BUILTIN, HIGH);
  delay(100);

  digitalWrite(LED_BUILTIN, LOW);
  delay(100);
  digitalWrite(LED_BUILTIN, HIGH);
  delay(100);

  digitalWrite(LED_BUILTIN, LOW);
  delay(600);
  digitalWrite(LED_BUILTIN, HIGH);
  delay(100);

  digitalWrite(LED_BUILTIN, LOW);
  delay(100);
  digitalWrite(LED_BUILTIN, HIGH);
}
