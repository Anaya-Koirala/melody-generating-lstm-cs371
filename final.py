# %% [markdown]
# # MelodyTron
# %% [markdown]
# ## Setup and Configuration
# %% [markdown]
# #### Imports
# %%
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Optional

import music21 as m21
import numpy as np
import tensorflow.keras as keras
# %% [markdown]
# #### Hyperparameters
# %%
RANDOM_SEED: int = 42

## LSTM training hyperparameters
SEQUENCE_LENGTH: int = 64
TIME_STEP: float = 0.25
BATCH_SIZE: int = 64
EPOCHS: int = 10
LEARNING_RATE: float = 0.002
LSTM_UNITS: int = 256
DROPOUT_RATE: float = 0.2
ACCEPTABLE_DURATIONS: list[float] = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]

# Generation hyperparameters
NUM_SEED_FILES: int = 10
SEED_SECONDS: float = 5.0
NUM_STEPS: int = 300
TEMPERATURE: float = 0.3
# %% [markdown]
# #### Paths
# %%
PROJECT_ROOT: Path = Path.cwd()

## MAESTRO Dataset
COMPOSER_NAME: str = "Frédéric Chopin"
MAESTRO_DIR: Path = PROJECT_ROOT / "maestro-v3.0.0"
MAESTRO_METADATA_CSV: Path = MAESTRO_DIR / "maestro-v3.0.0.csv"

## MIDI Location
MELODIES_DIR: Path = PROJECT_ROOT / "melodies"

## Output paths
RESULTS_DIR: Path = PROJECT_ROOT / "results"
MODEL_PATH: Path = PROJECT_ROOT / "model.keras"
MAPPING_PATH: Path = PROJECT_ROOT / "mapping.json"
SINGLE_FILE_DATASET: Path = PROJECT_ROOT / "file_dataset"

# Optional preprocessing toggle used by both sources.
APPLY_TRANSPOSITION: bool = False

MELODIES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# %% [markdown]
# # Preprocessing
# %%
def transpose(song: m21.stream.Stream) -> m21.stream.Stream:
    # We normalize to C major / A minor so the model focuses on melodic shape,
    # not key signature differences between songs.
    key = song.analyze("key")
    target_tonic = "C" if key.mode == "major" else "A"
    interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch(target_tonic))
    return song.transpose(interval)

def has_acceptable_durations(song: m21.stream.Stream, acceptable_durations: list[float]) -> bool:
    # We keep only files whose events can be represented by our token step size.
    for event in song.flatten().notesAndRests:
        if float(event.duration.quarterLength) not in acceptable_durations:
            return False
    return True
# %% [markdown]
# ## Extract melodies from MAESTRO dataset
# %%
def get_melody(score: m21.stream.Score) -> m21.stream.Part:
    """Extract a single melodic line using the fourth_hour right-hand heuristic."""
    events_by_offset: DefaultDict[float, list[tuple[m21.note.Note, int]]] = defaultdict(list)

    for el in score.flatten().notes:
        if isinstance(el, m21.note.Note):
            events_by_offset[float(el.offset)].append((el, int(el.pitch.midi)))
        elif isinstance(el, m21.chord.Chord):
            for p in el.pitches:
                pseudo = m21.note.Note(p)
                pseudo.quarterLength = float(el.quarterLength)
                pseudo.volume.velocity = el.volume.velocity
                events_by_offset[float(el.offset)].append((pseudo, int(p.midi)))

    melody = m21.stream.Part(id="RightHandMelody")
    for cls in (m21.tempo.MetronomeMark, m21.meter.TimeSignature, m21.key.KeySignature):
        for ctx in score.flatten().getElementsByClass(cls):
            melody.insert(float(ctx.offset), ctx)

    last_pitch: Optional[int] = None
    last_end: float = 0.0

    for offset in sorted(events_by_offset.keys()):
        candidates = events_by_offset[offset]
        right_hand = [c for c in candidates if c[1] >= 60] or candidates

        if last_pitch is None:
            selected_note, selected_pitch = max(right_hand, key=lambda x: x[1])
        else:
            selected_note, selected_pitch = min(
                right_hand,
                key=lambda x: (abs(x[1] - last_pitch), -x[1]),
            )

        onset = float(offset)
        duration = max(0.125, float(selected_note.quarterLength))

        if onset > last_end:
            melody.insert(last_end, m21.note.Rest(quarterLength=onset - last_end))

        out_note = m21.note.Note(selected_pitch)
        out_note.quarterLength = duration
        out_note.volume.velocity = selected_note.volume.velocity
        melody.insert(onset, out_note)

        last_pitch = selected_pitch
        last_end = max(last_end, onset + duration)

    return melody

def extract_melody_maestro():
    # Filter only the composer's files
    maestro_files = []
    with MAESTRO_METADATA_CSV.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if row["canonical_composer"] == COMPOSER_NAME:
                maestro_files.append(MAESTRO_DIR / row["midi_filename"])
    print(f"Found {len(maestro_files)} MIDI files for composer: {COMPOSER_NAME}")
    # Extract one melody line from each full piano MIDI and save to MELODIES_DIR.
    for i, midi_path in enumerate(maestro_files):
        try:
            score = m21.converter.parse(str(midi_path))
            melody = get_melody(score)
            if APPLY_TRANSPOSITION:
                melody = transpose(melody)
            # Save file as <original_name>_melody.midi
            out_path = MELODIES_DIR / f"{i:05d}_{midi_path.stem}_melody.midi"
            melody.write("midi", fp=str(out_path))
        except Exception as e:
            print(f"Error processing {midi_path.name}: {e}")


def iter_midi_files(root_dir: Path) -> list[Path]:
    """Return all MIDI files under a directory (supports .mid and .midi)."""
    return sorted(list(root_dir.rglob("*.midi")) + list(root_dir.rglob("*.mid")))
# %% [markdown]
# ## Tokenization + Dataset Building
# %%
def encode_midi_file_to_tokens(
    midi_path: Path,
    time_step: float = TIME_STEP,
    acceptable_durations: Optional[list[float]] = None,
) -> Optional[list[str]]:
    """Encode one MIDI to [pitch/r/_] tokens. Return None if duration filter fails."""
    song = m21.converter.parse(str(midi_path))

    if not has_acceptable_durations(song, acceptable_durations):
        return None

    tokens: list[str] = []
    for event in song.flatten().notesAndRests:
        symbol = str(int(event.pitch.midi)) if isinstance(event, m21.note.Note) else "r"
        steps = max(1, int(round(float(event.duration.quarterLength) / time_step)))
        tokens.append(symbol)
        tokens.extend(["_"] * (steps - 1))
    return tokens


def create_single_file_dataset(
    melody_root: Path,
    file_dataset_path: Path,
    sequence_length: int,
) -> str:
    """Concatenate all tokenized MIDI files into one stream with delimiter tokens."""
    delimiter = ["/"] * sequence_length
    all_tokens: list[str] = []
    skipped = 0

    for midi_path in iter_midi_files(melody_root):
        tokens = encode_midi_file_to_tokens(
            midi_path,
            time_step=TIME_STEP,
            acceptable_durations=ACCEPTABLE_DURATIONS,
        )
        if not tokens:
            skipped += 1
            continue
        all_tokens.extend(tokens)
        all_tokens.extend(delimiter)

    songs = " ".join(all_tokens)
    file_dataset_path.write_text(songs)
    print(f"Skipped files due to duration filter: {skipped}")
    return songs


def create_mapping(songs: str, mapping_path: Path) -> dict[str, int]:
    vocabulary = sorted(set(songs.split()))
    mapping = {token: idx for idx, token in enumerate(vocabulary)}
    with mapping_path.open("w", encoding="utf-8") as fp:
        json.dump(mapping, fp, indent=2)
    return mapping


def convert_songs_to_int(songs: str, mapping: dict[str, int]) -> list[int]:
    return [mapping[token] for token in songs.split()]


def generate_training_sequences(
    songs: str,
    mapping: dict[str, int],
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    int_songs = convert_songs_to_int(songs, mapping)
    vocab_size = len(mapping)
    num_sequences = len(int_songs) - sequence_length

    raw_inputs = [int_songs[i : i + sequence_length] for i in range(num_sequences)]
    targets = np.array([int_songs[i + sequence_length] for i in range(num_sequences)])
    inputs = keras.utils.to_categorical(raw_inputs, num_classes=vocab_size)

    print(f"Training sequences : {len(inputs):,}")
    print(f"Input shape        : {inputs.shape}")
    return inputs, targets


if not MELODIES_DIR.exists():
    raise ValueError(f"Melodies directory not found: {MELODIES_DIR}")

songs_str = create_single_file_dataset(
    MELODIES_DIR,
    SINGLE_FILE_DATASET,
    SEQUENCE_LENGTH,
)
if not songs_str.strip():
    raise ValueError("No tokens were produced. Check dataset source, paths, or duration filtering.")

mapping = create_mapping(songs_str, MAPPING_PATH)
inputs, targets = generate_training_sequences(songs_str, mapping, SEQUENCE_LENGTH)

print(f"Vocabulary size: {len(mapping)}")
print(f"Mapping file   : {MAPPING_PATH}")
# %% [markdown]
# ## 6) Model Build + Training (melody_lstm architecture)
# %%
def build_model(vocab_size: int) -> keras.Model:
    inputs_layer = keras.layers.Input(shape=(None, vocab_size))
    x = keras.layers.LSTM(LSTM_UNITS)(inputs_layer)
    x = keras.layers.Dropout(DROPOUT_RATE)(x)
    outputs_layer = keras.layers.Dense(vocab_size, activation="softmax")(x)

    model = keras.Model(inputs_layer, outputs_layer)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"],
    )
    return model

# Comment me if you want to just generate and not run
model = build_model(len(mapping))
# Comment me if you want to just generate and not run
model.summary()

stop_early_callback = keras.callbacks.EarlyStopping(
    monitor="loss",
    patience=3,
    restore_best_weights=True,
)

reduce_lr_plateau = keras.callbacks.ReduceLROnPlateau(
    monitor="loss",
    factor=0.5,
    patience=2,
    min_lr=1e-5,
)

# Comment me if you want to just generate and not run
history = model.fit(inputs,targets,epochs=EPOCHS,batch_size=BATCH_SIZE,callbacks=[stop_early_callback, reduce_lr_plateau])

# Comment me if you want to just generate and not run
model.save(str(MODEL_PATH))
# Comment me if you want to just generate and not run
print(f"Model saved to {MODEL_PATH}")
# %% [markdown]
# ## 7) Seed from MIDI First N Seconds + Melody Generation
# %%
class MelodyGenerator:
    """
    A wrapper around the trained LSTM model for generating new melodies.

    This class handles:
    - Loading the trained model from disk
    - Generating new token sequences autoregressively
    - Decoding tokens back into MIDI format
    """

    def __init__(self, model_path: Path, mapping_path: Path) -> None:
        """
        Initialize the MelodyGenerator by loading model and mapping files.

        Args:
            model_path: Path to the saved .keras model file
            mapping_path: Path to the mapping.json file
        """
        # Load the trained model from disk
        # The .keras format includes architecture, weights, and compilation config
        # No need for custom_objects - the .keras format handles all parameters correctly!
        self.model = keras.models.load_model(str(model_path))

        # Load the token → integer mapping from JSON file
        with mapping_path.open("r") as fp:
            self._mappings: dict[str, int] = json.load(fp)

        # Create reverse mapping: integer → token
        # This is used during generation to convert predicted integers back to strings
        self._id_to_token: dict[int, str] = {v: k for k, v in self._mappings.items()}

        # Starting symbols: SEQUENCE_LENGTH "/" tokens
        # This provides a clean context window for generation (no song content yet)
        self._start_symbols: list[str] = ["/"] * SEQUENCE_LENGTH

    def generate_melody(self,seed: list[str],num_steps: int = 500,max_sequence_length: int = SEQUENCE_LENGTH,temperature: float = 0.7,) -> list[str]:
        """
        Autoregressively generate a melody from a seed phrase.

        Algorithm:
        1. Start with seed tokens + start symbols as context
        2. Feed last max_sequence_length tokens to model
        3. Model predicts probability over vocabulary
        4. Sample next token using temperature scaling
        5. Add sampled token to context and repeat

        Args:
            seed: List of tokens to start with (e.g., ["67", "_", "_", "_"])
            num_steps: Maximum number of tokens to generate
            max_sequence_length: How many previous tokens to feed to model
            temperature: Controls randomness (0.3=conservative, 1.0=normal, 1.5=creative)

        Returns:
            List of generated tokens as strings
        """
        # Use the seed tokens directly (already a list)
        seed_tokens = seed

        # Initialize melody with seed tokens (we'll append generated tokens to this)
        melody = seed_tokens.copy()

        # Build the initial context: start symbols + seed tokens, converted to integers
        context = [self._mappings[t] for t in self._start_symbols + seed_tokens]

        # Generate new tokens one by one
        for _ in range(num_steps):
            # Get the last max_sequence_length tokens from context
            # This is what we feed to the model (the sliding window)
            window = context[-max_sequence_length:]

            # Convert integers to one-hot vectors for model input
            onehot = keras.utils.to_categorical(window, num_classes=len(self._mappings))

            # Add batch dimension: (seq_length, vocab_size) → (1, seq_length, vocab_size)
            onehot = onehot[np.newaxis, ...]

            # Get model's probability predictions for the next token
            # predictions shape: (1, vocab_size) - probabilities over all tokens
            probabilities = self.model.predict(onehot, verbose=0)[0]

            # Sample next token using temperature scaling
            next_id = self._sample_with_temperature(probabilities, temperature)

            # Convert the integer ID back to a token string
            next_token = self._id_to_token[next_id]

            # Add the sampled integer to context (for future predictions)
            context.append(next_id)

            # If we hit the delimiter token, stop generation
            if next_token == "/":
                break

            # Add the token to our generated melody
            melody.append(next_token)

        # Return the full list of generated tokens
        return melody


    @staticmethod
    def _sample_with_temperature(probabilities: np.ndarray, temperature: float) -> int:
        """
        Sample a token index using temperature-scaled probability distribution.

        Temperature scaling adjusts how "confident" (peaky) the distribution is:
        - Low temp (0.3): sharper distribution, pick best tokens often
        - High temp (1.5): flatter distribution, more random choices

        Formula: p'_i = exp(log(p_i) / T) / sum_j(exp(log(p_j) / T))

        Args:
            probabilities: Array of probabilities over vocabulary from model
            temperature: Scaling factor (>0)

        Returns:
            Sampled token index
        """
        # Compute log probabilities, scale by temperature
        # log(...) converts probabilities to log space; division by T scales the values
        log_probs = np.log(np.maximum(probabilities, 1e-10)) / max(temperature, 1e-6)

        # Subtract the maximum for numerical stability (prevents overflow)
        # This doesn't change the relative values, just shifts everything
        log_probs -= np.max(log_probs)

        # Convert back from log space: exp(log_probs) gives scaled probabilities
        dist = np.exp(log_probs)

        # Normalize so probabilities sum to 1
        dist /= dist.sum()

        # Sample one index from the distribution
        return int(np.random.choice(len(dist), p=dist))


    def save_melody(self,melody: list[str], file_name: Path | str, step_duration: float = TIME_STEP) -> None:
        """
        Convert a token sequence into a MIDI file.

        Process:
        1. Iterate through tokens
        2. When we see a note/rest followed by "_" tokens, expand the duration
        3. Create music21 Note/Rest objects with proper durations
        4. Write to MIDI file

        Args:
            melody: List of token strings (e.g., ["67", "_", "_", "65", ...])
            file_name: Output MIDI file path (string or Path object)
            step_duration: Duration of each time step in quarter lengths (0.25 = 16th note)
        """
        # Create an empty music21 Stream (container for musical events)
        stream = m21.stream.Stream()

        # Track the current note/rest being built
        start_symbol: Optional[str] = None

        # Count how many time steps (steps) the current note/rest spans
        step_counter: int = 1

        def _flush() -> None:
            """
            Helper function: finalize the current note/rest and add it to the stream.
            """
            # If we haven't started a note/rest yet, nothing to do
            if start_symbol is None:
                return

            # Calculate total duration: number of steps × step_duration
            # Example: 4 steps × 0.25 = 1.0 quarter length (quarter note)
            ql = step_duration * step_counter

            # Create either a Rest or a Note depending on the symbol
            if start_symbol == "r":
                # "r" means rest
                event = m21.note.Rest(quarterLength=ql)
            else:
                # Otherwise it's a MIDI pitch number (convert to int)
                event = m21.note.Note(int(start_symbol), quarterLength=ql)

            # Add the note/rest to the stream
            stream.append(event)

        # Process each token in the melody
        for i, symbol in enumerate(melody):
            # Check if this is a continuation token "_" AND we're not at the end
            if symbol != "_" or i + 1 == len(melody):
                # This is a new note/rest, so flush the previous one
                _flush()

                # Start a new note/rest with this symbol
                start_symbol = symbol

                # Reset step counter for the new note/rest
                step_counter = 1
            else:
                # This is a continuation token, extend the current note/rest
                step_counter += 1

        # Write the stream to a MIDI file
        stream.write("midi", str(file_name))
        print(f"Melody saved to {file_name}")
# %%
def extract_seed_tokens_from_midi(midi_path: Path,seed_seconds: float,time_step: float = TIME_STEP,) -> list[str]:
    """Extract seed tokens from the first N seconds of a MIDI file."""
    score = m21.converter.parse(str(midi_path))
    tokens: list[str] = []

    elapsed = 0.0
    for event in score.flatten().notesAndRests:
        event_seconds = event.seconds if event.seconds is not None else 0.0
        event_seconds = float(event_seconds)

        # Fallback if seconds context is unavailable in parsed stream.
        if event_seconds <= 0:
            event_seconds = float(event.duration.quarterLength)

        if elapsed >= seed_seconds:
            break

        symbol = str(int(event.pitch.midi)) if isinstance(event, m21.note.Note) else "r"

        if elapsed + event_seconds <= seed_seconds:
            steps = max(1, int(round(float(event.duration.quarterLength) / time_step)))
            tokens.append(symbol)
            tokens.extend(["_"] * (steps - 1))
            elapsed += event_seconds
        else:
            remaining = max(0.0, seed_seconds - elapsed)
            frac = remaining / max(event_seconds, 1e-8)
            clipped_q = float(event.duration.quarterLength) * frac
            steps = max(1, int(round(clipped_q / time_step)))
            tokens.append(symbol)
            tokens.extend(["_"] * (steps - 1))
            break

    return tokens


def choose_seed_random(num_files: int, seed: int) -> list[Path]:
    # Seed files are sampled from the same directory used for training data.
    candidates = iter_midi_files(MELODIES_DIR)
    if not candidates:
        raise ValueError(f"No MIDI files found under {MELODIES_DIR}")
    rng = np.random.default_rng(seed)
    n_selected = min(num_files, len(candidates))
    return list(rng.choice(candidates, size=n_selected, replace=False))

def choose_seed_file(file_path: Path) -> list[Path]:
    if not file_path.exists():
        raise ValueError(f"Specified seed file not found: {file_path}")
    return [file_path]
# %% [markdown]
# #### Randomly choosen seed files from training set
# %%

mg = MelodyGenerator(MODEL_PATH, MAPPING_PATH)
selected_seed_files = choose_seed_random(NUM_SEED_FILES, RANDOM_SEED)

for seed_midi_path in selected_seed_files:
    seed_tokens = extract_seed_tokens_from_midi(
        midi_path=seed_midi_path,
        seed_seconds=SEED_SECONDS,
        time_step=TIME_STEP,
    )

    melody = mg.generate_melody(
        seed=seed_tokens,
        num_steps=NUM_STEPS,
        max_sequence_length=SEQUENCE_LENGTH,
        temperature=TEMPERATURE,
    )

    output_path = RESULTS_DIR / f"{seed_midi_path.stem}_generated.mid"
    mg.save_melody(melody, file_name=output_path, step_duration=TIME_STEP)

    print(f"Seed file          : {seed_midi_path.name}")
    print(f"Seed tokens        : {len(seed_tokens)}")
    print(f"Generated tokens   : {len(melody)}")
    print(f"Output MIDI        : {output_path}")
# %% [markdown]
# #### User-specified seed file
# %%

mg = MelodyGenerator(MODEL_PATH, MAPPING_PATH)
selected_seed_files = choose_seed_file(Path("/Accounts/turing/students/s28/koiran01/Documents/cs371/fourth-hour/happy_birthday.mid"))

for seed_midi_path in selected_seed_files:
    seed_tokens = extract_seed_tokens_from_midi(
        midi_path=seed_midi_path,
        seed_seconds=SEED_SECONDS,
        time_step=TIME_STEP,
    )

    melody = mg.generate_melody(
        seed=seed_tokens,
        num_steps=NUM_STEPS,
        max_sequence_length=SEQUENCE_LENGTH,
        temperature=TEMPERATURE,
    )

    output_path = RESULTS_DIR / f"{seed_midi_path.stem}_generated.midi"
    mg.save_melody(melody, file_name=output_path, step_duration=TIME_STEP)

    print(f"Seed file          : {seed_midi_path.name}")
    print(f"Seed tokens        : {len(seed_tokens)}")
    print(f"Generated tokens   : {len(melody)}")
    print(f"Output MIDI        : {output_path}")
