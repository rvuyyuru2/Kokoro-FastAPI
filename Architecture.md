# Kokoro TTS Architecture

## Status

### Completed ✅
1. Core Module Structure
- Pipeline module with strategy pattern
  - Streaming strategy
  - Whole file strategy
  - Plugin support
- Inference module with CPU/GPU backends
  - ONNX optimization
  - PyTorch memory management
  - Model caching
- Text processing module with clean separation
  - Normalization
  - Phonemization
  - Tokenization
- Audio processing module with streaming
  - Format conversion
  - Normalization
  - Post-processing
  - Padding

2. API Integration
- Core TTS endpoints
  - Complete file generation
  - Streaming generation
  - Voice management
- OpenAI-compatible endpoints
  - Speech generation
  - Voice listing
  - Voice combination
- Error handling
  - Input validation
  - Error responses
  - Logging

3. Pipeline Implementation
- Base pipeline with strategy pattern
- Streaming strategy
  - Chunk processing
  - Memory efficiency
  - Client disconnect handling
- Whole file strategy
  - Complete file generation
  - Chunk combination
  - Post-processing

### Next Up 🎯
1. Testing Implementation
- [ ] Unit tests for modules
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Plugin tests

## Module Details

### 1. Pipeline (`api/src/pipeline/`) ✅
```python
# Strategy pattern
class GenerationStrategy(Protocol):
    def generate(
        self,
        pipeline: Pipeline,
        text: str,
        voice: str,
        speed: float = 1.0,
        format: str = "wav"
    ) -> Union[bytes, Iterator[bytes]]

# Factory functions
def create_streaming_pipeline(voices_dir: str = "voices") -> Pipeline
def create_whole_file_pipeline(voices_dir: str = "voices") -> Pipeline

# Convenience function
def process_text(
    text: str,
    voice: str,
    speed: float = 1.0,
    format: str = "wav",
    stream: bool = False
) -> Union[bytes, Iterator[bytes]]
```

### 2. Service Layer (`api/src/services/`) ✅
```python
class TTSService:
    async def generate_audio(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        format: str = "wav"
    ) -> bytes:
        """Generate complete audio file."""

    async def generate_stream(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        format: str = "wav"
    ) -> AsyncIterator[bytes]:
        """Generate streaming audio."""
```

### 3. API Endpoints (`api/src/routers/`) ✅
```python
# Core TTS endpoints
@router.post("/tts")
async def text_to_speech(
    text: str,
    voice: str,
    speed: float = 1.0,
    format: str = "wav"
)

@router.post("/tts/stream")
async def stream_tts(
    text: str,
    voice: str,
    speed: float = 1.0,
    format: str = "wav"
)

# OpenAI-compatible endpoints
@router.post("/audio/speech")
async def create_speech(
    request: OpenAISpeechRequest,
    client_request: Request
)
```

## Plugin System (`api/src/plugins/`) ✅
```python
class TTSHookSpec:
    @hookspec
    def pre_process_text(text: str) -> str
    def post_process_text(text: str) -> str
    def pre_process_audio(audio: np.ndarray) -> np.ndarray
    def post_process_audio(audio: np.ndarray) -> np.ndarray
    def customize_pipeline(pipeline: Pipeline) -> Pipeline
```

## Next Steps
1. Implement comprehensive tests:
   - Unit tests for each module
   - Integration tests
   - Performance benchmarks
   - Plugin system tests
2. Update documentation
3. Performance optimization