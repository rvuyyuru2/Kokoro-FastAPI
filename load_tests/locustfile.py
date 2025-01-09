import json
import random
import time
from locust import HttpUser, task, between, events
from pathlib import Path

class TTSUser(HttpUser):
    """User class for TTS load testing"""
    wait_time = between(1, 3)
    voices = []
    phrases = []

    def on_start(self):
        # Load phrases
        phrases_file = Path(__file__).parent / "sample_data" / "phrases.json"
        with open(phrases_file) as f:
            self.phrases = json.load(f)["phrases"]

        # Get available voices
        response = self.client.get("/v1/audio/voices")
        if response.status_code == 200:
            self.voices = response.json()["voices"]
        else:
            self.voices = ["af_bella"]  # Fallback to a default voice

    def handle_streamed_response(self, response):
        """Process streaming response"""
        start_time = time.time()
        first_chunk = True
        
        for chunk in response.iter_content(chunk_size=8192):
            if chunk and first_chunk:
                # Record time to first chunk
                first_chunk_time = int((time.time() - start_time) * 1000)
                events.request.fire(
                    request_type="First Chunk",
                    name="Time to First Chunk",
                    response_time=first_chunk_time,
                    response_length=0,
                    exception=None,
                    context={}
                )
                first_chunk = False

    def generate_text(self, target_length):
        """Generate text by combining phrases to reach approximate target length"""
        text = []
        current_length = 0
        
        # First phrase
        phrase = random.choice(self.phrases)
        text.append(phrase)
        current_length = len(phrase)
        
        # Add phrases until we get close to target length
        while current_length < target_length:
            phrase = random.choice(self.phrases)
            # Don't exceed target length by too much
            if current_length + len(phrase) > target_length * 1.2:
                break
            text.append(phrase)
            current_length += len(phrase)
        
        return " ".join(text)

    @task(3)
    def test_short_stream(self):
        """Test short streaming responses (50-200 chars)"""
        text_length = random.randint(50, 200)
        with self.client.post(
            "/v1/audio/speech",
            json={
                "input": self.generate_text(text_length),
                "voice": random.choice(self.voices),
                "response_format": "mp3",
                "speed": 1.0,
                "stream": True
            },
            stream=True,
            catch_response=True
        ) as response:
            try:
                self.handle_streamed_response(response)
                response.success()
            except Exception as e:
                response.failure(str(e))

    @task(2)
    def test_medium_stream(self):
        """Test medium streaming responses (200-500 chars)"""
        text_length = random.randint(200, 500)
        with self.client.post(
            "/v1/audio/speech",
            json={
                "input": self.generate_text(text_length),
                "voice": random.choice(self.voices),
                "response_format": "mp3",
                "speed": 1.0,
                "stream": True
            },
            stream=True,
            catch_response=True
        ) as response:
            try:
                self.handle_streamed_response(response)
                response.success()
            except Exception as e:
                response.failure(str(e))

    # @task(1)
    # def test_long_stream(self):
    #     """Test long streaming responses (500-1000 chars)"""
    #     text_length = random.randint(500, 1000)
    #     with self.client.post(
    #         "/v1/audio/speech",
    #         json={
    #             "input": self.generate_text(text_length),
    #             "voice": random.choice(self.voices),
    #             "response_format": "mp3",
    #             "speed": 1.0,
    #             "stream": True
    #         },
    #         stream=True,
    #         catch_response=True
    #     ) as response:
    #         try:
    #             self.handle_streamed_response(response)
    #             response.success()
    #         except Exception as e:
    #             response.failure(str(e))

    # @task(2)
    # def get_voices(self):
    #     """Occasionally refresh available voices"""
    #     self.client.get("/v1/audio/voices")
