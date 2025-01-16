# Kokoro TTS Architecture

## Core Concepts

Kokoro TTS uses a plugin-based architecture with three main processing stages:

1. Text Processing
2. Voice Style Generation
3. Audio Generation

Each stage uses a strategy pattern that allows for easy extension and customization.

## Strategy System

Each major component follows this structure:

```
component/
├── strategy_interface.py    # Base interface
├── strategy_factory.py      # Strategy management
└── implementations/        # Individual strategies
    ├── strategy_a/
    └── strategy_b/
```

This allows us to:
- Add new implementations without changing existing code
- Configure and combine strategies flexibly
- Test components in isolation

## Text Processing Pipeline

Text goes through three main stages, each with pluggable strategies:

```python
# Example pipeline configuration
pipeline = TextProcessingPipeline([
    chunking_strategy,      # Split text into manageable chunks
    phonemize_strategy,     # Convert text to phonemes
    tokenize_strategy       # Convert phonemes to model tokens
])

# Usage
result = await pipeline.process("Hello world")
```

### Chunking Strategies
- **Static**: Fixed-size chunks with basic boundary detection
- **Dynamic**: Smart chunking using prosody and semantic analysis
- **Future**: ML-based chunking using sentence embeddings

### Phonemization Strategies
- **Espeak**: Current implementation using espeak-ng
- **Future**: Language-specific phonemizers
- **Future**: Custom pronunciation rules engine

### Tokenization Strategies
- **Basic**: Current vocabulary-based implementation
- **Future**: BPE (Byte Pair Encoding)
- **Future**: SentencePiece support

## Multi-Language Support

Language support is implemented through strategy configuration:

```python
# Configure for Japanese
jp_pipeline = TextProcessingPipeline([
    ChunkingStrategy(
        language="ja",
        rules=japanese_chunking_rules
    ),
    PhonemeStrategy(
        language="ja",
        phoneme_set=japanese_phonemes
    ),
    TokenizeStrategy(
        vocab=japanese_vocab
    )
])

# Configure for English
en_pipeline = TextProcessingPipeline([
    ChunkingStrategy(
        language="en",
        rules=english_chunking_rules
    ),
    PhonemeStrategy(
        language="en",
        phoneme_set=english_phonemes
    ),
    TokenizeStrategy(
        vocab=english_vocab
    )
])
```

### Language-Specific Components

Each language can have specialized components:

1. **Chunking Rules**:
   - Japanese: Split on 。、
   - English: Split on .!?
   - Chinese: Character-based splitting

2. **Phoneme Sets**:
   - Japanese: Hiragana-based
   - English: IPA-based
   - Chinese: Pinyin-based

3. **Tokenization**:
   - Japanese: Character-based
   - English: Subword-based
   - Chinese: Character-based

## Implementation Example

Here's how it all comes together:

```python
# Configure strategies
chunker = factory.get_strategy("dynamic", 
    language="ja",
    boundary_weights={
        'sentence': 1.0,
        'phrase': 0.8,
        'word': 0.6
    }
)

phonemizer = factory.get_strategy("japanese",
    use_accent=True,
    preserve_pitch=True
)

tokenizer = factory.get_strategy("bpe",
    vocab_size=8000,
    model_path="ja_tokenizer.model"
)

# Create pipeline
pipeline = TextProcessingPipeline([
    chunker,
    phonemizer,
    tokenizer
])

# Process text
async def generate_speech(text: str, voice: str):
    # Process through pipeline
    tokens = await pipeline.process(text)
    
    # Generate audio
    audio = await tts_service.generate_audio(
        tokens=tokens,
        voice=voice
    )
    
    return audio
```

## Adding New Languages

To add support for a new language:

1. Create language-specific strategies:
   ```python
   class ThaiChunkStrategy(ChunkingStrategy):
       """Thai-specific text chunking"""
       
   class ThaiPhonemeStrategy(PhonemeStrategy):
       """Thai phoneme conversion"""
   ```

2. Register with factories:
   ```python
   chunking_factory.register_strategy("thai", ThaiChunkStrategy)
   phoneme_factory.register_strategy("thai", ThaiPhonemeStrategy)
   ```

3. Configure pipeline:
   ```python
   thai_pipeline = TextProcessingPipeline([
       ThaiChunkStrategy(),
       ThaiPhonemeStrategy(),
       TokenizeStrategy(vocab=thai_vocab)
   ])
   ```

## Benefits

This architecture provides:

1. **Flexibility**: Easy to add new languages and strategies
2. **Maintainability**: Components are isolated and testable
3. **Performance**: Can optimize each stage independently
4. **Extensibility**: Plugin system for community contributions