# %% [markdown]
# # MelodyTron
# 
# - A simple LSTM that learns to continue piano melodies.
# - We train it on melodies lines from the [KernScores](https://kern.ccarh.org) dataset, which is a collection of German folk music.
# - We generate new music starting from a given melody (the seed).
# 
# <span style="color:green"> The KernScores dataset has the files in the ``.krn`` format. MIDI ``.midi`` is the standard, hence we have a script ``krn_to_midi.py`` to convert all the dataset into the MIDI format</span>
# 
# #### Sources:
# - [Music21](https://web.mit.edu/music21/) for MIDI parsing and manipulation.
# - [TensorFlow Keras](https://www.tensorflow.org/api_docs/python/tf/keras) for building and training the LSTM model.
# - [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Chris Olah on how LSTMs work.
# - [Melody generation with RNN-LSTM](https://youtube.com/playlist?list=PL-wATfeyAMNr0KMutwtbeDCmpwvtul-Xz&si=XQ4HDloiU1V0313H) YouTube series by researcher [Valerio Velardo](https://valeriovelardoadvisor.com/).
# %% [markdown]
# ## The MIDI Protocol
# The MIDI protocol encodes musical information as a series of events, each with specific attributes.
# For our purposes, we focus on the following key variables:
# | Variable | Meaning |
# |----------|---------|
# | `offset` | Start time of a note, in quarter lengths |
# | `quarterLength` | Duration of a note in quarter notes (standard MIDI unit) |
# | `midi` (pitch number) | 0 to 127 scale where 60 is middle C; higher numbers are higher pitches |
# | `velocity` | How hard the note was struck (0 to 127); kept to preserve dynamics |
# %% [markdown]
# ## Setup and Configuration
# 
# - We set up the libraries, hyperparameters, and file paths that the rest of the notebook depends on.
# - The hyperparameters are for training the LSTM and for generation
# - The paths point to the dataset and location of generated files
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
# Fix the random seed so file selection is reproducible.
RANDOM_SEED: int = 42

## LSTM training hyperparameters
# Set the number of time steps the model sees at once.
SEQUENCE_LENGTH: int = 64
# Set the token granularity to a sixteenth-note grid.
TIME_STEP: float = 0.25
# Set how many training examples each update processes together.
BATCH_SIZE: int = 64
# Set how many full training passes to run.
EPOCHS: int = 5
# Set the learning rate used by the Adam optimizer.
LEARNING_RATE: float = 0.001
# Set the number of hidden units in the LSTM layer.
LSTM_UNITS: int = 256
# Set the dropout rate used to reduce overfitting.
DROPOUT_RATE: float = 0.2
# Restrict events to durations that fit the token grid.
ACCEPTABLE_DURATIONS: list[float] = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]

# Generation hyperparameters
NUM_SEED_FILES: int = 10
SEED_SECONDS: float = 5.0
# Set the maximum number of new tokens to generate.
NUM_STEPS: int = 1000
TEMPERATURE: float = 0.7
# %% [markdown]
# #### Paths
# %%
PROJECT_ROOT: Path = Path.cwd()

## MAESTRO Dataset
COMPOSER_NAME: str = "Frédéric Chopin"
MAESTRO_DIR: Path = PROJECT_ROOT / "maestro-v3.0.0"
MAESTRO_METADATA_CSV: Path = MAESTRO_DIR / "maestro-v3.0.0.csv"

MELODIES_DIR: Path = PROJECT_ROOT / "melodies"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
MODEL_PATH: Path = PROJECT_ROOT / "model.keras"
MAPPING_PATH: Path = PROJECT_ROOT / "mapping.json"
SINGLE_FILE_DATASET: Path = PROJECT_ROOT / "file_dataset"

APPLY_TRANSPOSITION: bool = False

MELODIES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# %% [markdown]
# # Preprocessing
# This section handles two preprocessing steps:
# - Transposing every piece into a common key.
# - Filtering out files with note durations that don't fit our token grid.
# 
# Transposition is useful to reduce the number of unique notes that the model has to learn. It normalizes all possible notes in the piano range down to the white keys.
# %%
def transpose(song: m21.stream.Stream) -> m21.stream.Stream:
    # We normalize to C major / A minor
    key = song.analyze("key")
    target_tonic = "C" if key.mode == "major" else "A"
    interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch(target_tonic))
    return song.transpose(interval)

def has_acceptable_durations(song: m21.stream.Stream, acceptable_durations: list[float]) -> bool:
    for event in song.flatten().notesAndRests:
        if float(event.duration.quarterLength) not in acceptable_durations:
            return False
    return True
# %% [markdown]
# ## Extract melodies from MAESTRO dataset
# 
# - MAESTRO gives us full two-hand piano performances, but we want to extract only the single melodic line.
# - This section pulls the right-hand melody from a given composer (given above) and saves it into new MIDI files.
# -  For each moment in time:
#     1. Look at the notes being played, prefer the ones at or above middle C.
#     2. Pick the one that is the highest value, and keeps the melodic line moving smoothly.
#     3. Chords are split into their individual pitches and one of them can be chosen to preserve musical info.
#     4. Preserve the tempo, time signature, and key signature from the original score so the saved melody still plays back at the right speed and feel.
#     5. (Optionally) Transpose the piece
# %% [markdown]
# #### Note on the MAESTRO dataset
# - We initially planned to use the [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro), which contains classical piano performance by various composers.
# - We extracted only the melodies (right-hand part) to train the model on single-note sequences.
# - Ultimately, the generated output wasn't good, because the melodies in MAESTRO are more complex and less consistent than those in KernScores, which made it harder for the model to learn clear patterns.
# %%
def get_melody(score: m21.stream.Score) -> m21.stream.Part:
    # Store the annotated value for later use.
    events_by_offset: DefaultDict[float, list[tuple[m21.note.Note, int]]] = defaultdict(list)

    # Walk through the flattened score and collect each note or chord.
    for el in score.flatten().notes:
        if isinstance(el, m21.note.Note):
            events_by_offset[float(el.offset)].append((el, int(el.pitch.midi)))
        elif isinstance(el, m21.chord.Chord):
            for p in el.pitches:
                pseudo = m21.note.Note(p)
                pseudo.quarterLength = float(el.quarterLength)
                pseudo.volume.velocity = el.volume.velocity
                events_by_offset[float(el.offset)].append((pseudo, int(p.midi)))

    # Start the output token list with the seed melody.
    melody = m21.stream.Part(id="RightHandMelody")
    for cls in (m21.tempo.MetronomeMark, m21.meter.TimeSignature, m21.key.KeySignature):
        # Copy each tempo, meter, or key event into the output part.
        for ctx in score.flatten().getElementsByClass(cls):
            # Insert the current event at the correct offset in the score.
            melody.insert(float(ctx.offset), ctx)

    # Store the annotated value for later use.
    last_pitch: Optional[int] = None
    # Store the annotated value for later use.
    last_end: float = 0.0

    # Process each onset in chronological order.
    for offset in sorted(events_by_offset.keys()):
        # Store the computed value for the next step in the pipeline.
        candidates = events_by_offset[offset]
        # Store the computed value for the next step in the pipeline.
        right_hand = [c for c in candidates if c[1] >= 60] or candidates

        # Handle the first melody note separately from later ones.
        if last_pitch is None:
            # Store the computed value for the next step in the pipeline.
            selected_note, selected_pitch = max(right_hand, key=lambda x: x[1])
        else:
            # Store the computed value for the next step in the pipeline.
            selected_note, selected_pitch = min(
                right_hand,
                key=lambda x: (abs(x[1] - last_pitch), -x[1]),
            )

        # Store the computed value for the next step in the pipeline.
        onset = float(offset)
        # Store the computed value for the next step in the pipeline.
        duration = max(0.125, float(selected_note.quarterLength))

        # Insert a rest whenever there is a gap before the next note.
        if onset > last_end:
            # Insert the current event at the correct offset in the score.
            melody.insert(last_end, m21.note.Rest(quarterLength=onset - last_end))

        out_note = m21.note.Note(selected_pitch)
        out_note.quarterLength = duration
        out_note.volume.velocity = selected_note.volume.velocity
        melody.insert(onset, out_note)

        # Store the computed value for the next step in the pipeline.
        last_pitch = selected_pitch
        # Store the computed value for the next step in the pipeline.
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


# Define a helper that finds every MIDI file under a directory.
def iter_midi_files(root_dir: Path) -> list[Path]:
    """Return all MIDI files under a directory (supports .mid and .midi)."""
    return sorted(list(root_dir.rglob("*.midi")) + list(root_dir.rglob("*.mid")))
# %% [markdown]
# #### Token Format
# 
# Each melody is turned into a sequence of tokens before training. We use three kinds of symbols:
# 
# - **MIDI pitch numbers** like `60` or `67` mark the start of a note.
# - **`r`** marks the start of a rest.
# - **`_`** is a continuation symbol, meaning the previous note or rest is still going.
# 
# - We pick a time step of a sixteenth note and break every event into that many slots.
# - A half note at pitch 60 becomes `60 _ _ _ _ _ _ _`.
# - Files are joined with `/` delimiters so the model can tell where one melody ends and the next begins.
# 
# #### Hyperparameters
# 
# | Constant | Value | What it controls |
# |----------|-------|------------------|
# | `TIME_STEP` | 0.25 | One token equals a sixteenth note |
# | `SEQUENCE_LENGTH` | 64 | How many past tokens the model sees |
# | `BATCH_SIZE` | 128 | Sequences per training step |
# | `EPOCHS` | 5 | Full passes over the training data |
# 
# %% [markdown]
# ## Tokenization + Dataset Building
# 
# #### Token Format
# 
# Each melody is turned into a sequence of tokens before training. We use three kinds of symbols:
# 
# - **MIDI pitch numbers** like `60` or `67` mark the start of a note.
# - **`r`** marks the start of a rest.
# - **`_`** is a continuation symbol, meaning the previous note or rest is still going.
# 
# - We pick a time step of a sixteenth note and break every event into that many slots.
# - A half note at pitch 60 becomes `60 _ _ _ _ _ _ _`.
# - Files are joined with `/` delimiters so the model can tell where one melody ends and the next begins.
# - Files that contain durations outside our acceptable list are skipped, since they can't be cleanly represented on the sixteenth-note grid.
# 
# #### Hyperparameters
# 
# | Constant | Value | What it controls |
# |----------|-------|------------------|
# | `TIME_STEP` | 0.25 | One token equals a sixteenth note |
# | `SEQUENCE_LENGTH` | 64 | How many past tokens the model sees |
# | `BATCH_SIZE` | 128 | Sequences per training step |
# | `EPOCHS` | 5 | Full passes over the training data |
# 
# - Training examples come from sliding a window across the integer stream. For each position, the first `SEQUENCE_LENGTH` tokens are the input and the next token is the target. The inputs get one-hot encoded into shape `[SEQUENCE_LENGTH, vocab_size]`, which is what the LSTM expects.
# %%
# Define a helper that turns one MIDI file into pitch, rest, and continuation tokens.
def encode_midi_file_to_tokens(
    midi_path: Path,
    time_step: float = TIME_STEP,
    acceptable_durations: Optional[list[float]] = None,
) -> Optional[list[str]]:
    """Encode one MIDI to [pitch/r/_] tokens. Return None if duration filter fails."""
    # Parse the current MIDI file into a music21 score.
    song = m21.converter.parse(str(midi_path))

    # Check the current condition before continuing the pipeline.
    if not has_acceptable_durations(song, acceptable_durations):
        # Return the computed result from this helper.
        return None

    # Store the annotated value for later use.
    tokens: list[str] = []
    # Iterate through the current collection.
    for event in song.flatten().notesAndRests:
        # Store the computed value for the next step in the pipeline.
        symbol = str(int(event.pitch.midi)) if isinstance(event, m21.note.Note) else "r"
        # Store the computed value for the next step in the pipeline.
        steps = max(1, int(round(float(event.duration.quarterLength) / time_step)))
        # Append the current item to the growing collection.
        tokens.append(symbol)
        # Extend the current collection with additional tokens.
        tokens.extend(["_"] * (steps - 1))
    # Return the token sequence that was just built.
    return tokens


# Define a helper that concatenates all tokenized melodies into one training corpus.
def create_single_file_dataset(
    melody_root: Path,
    file_dataset_path: Path,
    sequence_length: int,
) -> str:
    """Concatenate all tokenized MIDI files into one stream with delimiter tokens."""
    # Store the computed value for the next step in the pipeline.
    delimiter = ["/"] * sequence_length
    # Store the annotated value for later use.
    all_tokens: list[str] = []
    # Store the computed value for the next step in the pipeline.
    skipped = 0

    # Process every MIDI file found under the melody directory.
    for midi_path in iter_midi_files(melody_root):
        # Collect the tokenized version of the current melody.
        tokens = encode_midi_file_to_tokens(
            midi_path,
            time_step=TIME_STEP,
            acceptable_durations=ACCEPTABLE_DURATIONS,
        )
        # Skip files that could not be tokenized cleanly.
        if not tokens:
            skipped += 1
            continue
        # Extend the current collection with additional tokens.
        all_tokens.extend(tokens)
        # Extend the current collection with additional tokens.
        all_tokens.extend(delimiter)

    # Store the computed value for the next step in the pipeline.
    songs = " ".join(all_tokens)
    file_dataset_path.write_text(songs)
    print(f"Skipped files due to duration filter: {skipped}")
    return songs


# Define a helper that builds and saves the token-to-id vocabulary.
def create_mapping(songs: str, mapping_path: Path) -> dict[str, int]:
    # Store the computed value for the next step in the pipeline.
    vocabulary = sorted(set(songs.split()))
    # Build the token-to-id mapping.
    mapping = {token: idx for idx, token in enumerate(vocabulary)}
    with mapping_path.open("w", encoding="utf-8") as fp:
        # Save the vocabulary mapping as JSON for later reuse.
        json.dump(mapping, fp, indent=2)
    return mapping


# Define a helper that converts token strings into integer ids.
def convert_songs_to_int(songs: str, mapping: dict[str, int]) -> list[int]:
    # Return the requested path in list form.
    return [mapping[token] for token in songs.split()]


# Define a helper that slices the token corpus into LSTM training windows.
def generate_training_sequences(
    songs: str,
    mapping: dict[str, int],
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Store the computed value for the next step in the pipeline.
    int_songs = convert_songs_to_int(songs, mapping)
    # Store the computed value for the next step in the pipeline.
    vocab_size = len(mapping)
    # Store the computed value for the next step in the pipeline.
    num_sequences = len(int_songs) - sequence_length

    # Store the computed value for the next step in the pipeline.
    raw_inputs = [int_songs[i : i + sequence_length] for i in range(num_sequences)]
    # Store the computed value for the next step in the pipeline.
    targets = np.array([int_songs[i + sequence_length] for i in range(num_sequences)])
    # Store the computed value for the next step in the pipeline.
    inputs = keras.utils.to_categorical(raw_inputs, num_classes=vocab_size)

    print(f"Training sequences : {len(inputs):,}")
    print(f"Input shape        : {inputs.shape}")
    return inputs, targets


# Fail early if the melodies directory is missing.
if not MELODIES_DIR.exists():
    # Raise a clear error so the notebook fails fast when data is missing.
    raise ValueError(f"Melodies directory not found: {MELODIES_DIR}")

songs_str = create_single_file_dataset(
    MELODIES_DIR,
    SINGLE_FILE_DATASET,
    SEQUENCE_LENGTH,
)

if not songs_str.strip():
    raise ValueError("No tokens were produced. Check dataset source, paths, or duration filtering.")

# Build the token-to-id mapping.
mapping = create_mapping(songs_str, MAPPING_PATH)

# Store the computed value for the next step in the pipeline.
inputs, targets = generate_training_sequences(songs_str, mapping, SEQUENCE_LENGTH)

print(f"Vocabulary size: {len(mapping)}")
print(f"Mapping file   : {MAPPING_PATH}")
# %% [markdown]
# ## Model Build + Training
# 
# - This is where the actual learning happens. We define a small LSTM, compile it, and train it to predict the next token given the previous 64.
# 
# - Inputs come in as one-hot vectors of shape `[sequence_length, vocab_size]` and pass through a single LSTM layer with 256 units, followed by dropout at 0.2 to cut down on overfitting. The output is a dense softmax over the vocabulary, giving a probability for each possible next token. We use Adam as the optimizer at a learning rate of 0.002 and sparse categorical crossentropy as the loss, since the targets are stored as plain integer ids.
# 
# - Two callbacks help training behave. `EarlyStopping` watches the loss and stops training if it stalls for three epochs in a row, restoring the best weights it saw. `ReduceLROnPlateau` cuts the learning rate in half when the loss plateaus, which lets the model keep making progress once the bigger steps stop helping. Once training finishes, the model is saved to `model.keras` so the generation step can load it later without having to retrain.
# %%
def build_model(vocab_size: int) -> keras.Model:
    # Define the input layer for one-hot encoded sequences.
    inputs_layer = keras.layers.Input(shape=(None, vocab_size))
    # Store the computed value for the next step in the pipeline.
    x = keras.layers.LSTM(LSTM_UNITS)(inputs_layer)
    # Store the computed value for the next step in the pipeline.
    x = keras.layers.Dropout(DROPOUT_RATE)(x)
    # Define the output layer that predicts the next token.
    outputs_layer = keras.layers.Dense(vocab_size, activation="softmax")(x)

    # Build or load the Keras model used for melody generation.
    model = keras.Model(inputs_layer, outputs_layer)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"],
    )
    return model

# Comment me to just generate
model = build_model(len(mapping))
# Comment me to just generate
model.summary()

# Store the computed value for the next step in the pipeline.
stop_early_callback = keras.callbacks.EarlyStopping(
    monitor="loss",
    patience=3,
    restore_best_weights=True,
)

# Store the computed value for the next step in the pipeline.
reduce_lr_plateau = keras.callbacks.ReduceLROnPlateau(
    monitor="loss",
    factor=0.5,
    patience=2,
    min_lr=1e-5,
)

# Comment me to just generate
# Capture the training history returned by Keras.
history = model.fit(
    inputs,
    targets,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[stop_early_callback, reduce_lr_plateau]
)
# Comment me to just generate
model.save(str(MODEL_PATH))
print(f"Model saved to {MODEL_PATH}")
# %% [markdown]
# ## Seed from MIDI First N Seconds + Melody Generation
# 
# - Now that the model is trained, we can use it to generate new melodies. The idea is to give the model a short opening from a real piece (the seed) and let it continue from there, one token at a time, until it either hits a delimiter or runs out of steps.
# 
# - The `MelodyGenerator` class wraps everything generation-related: loading the trained model and the token mapping, running the autoregressive loop, sampling tokens with temperature, and converting the resulting token sequence back into a playable MIDI file.
# 
# - Seeds come from the first few seconds of an existing MIDI file. We parse the file, walk through its notes and rests, and stop once we've collected enough material to use as a prompt. If an event would push us past the seed length, we trim it down so the prompt ends cleanly at the right time.
# 
# - Generation itself is autoregressive. The seed is fed in as starting context, the model predicts a probability distribution over the vocabulary, we sample one token from it, append the token to the context, and repeat. Sampling uses temperature scaling so the output isn't always the safest choice. Lower temperatures push the model toward its most confident predictions, while higher ones let less likely tokens through and produce more varied results. When the model predicts the `/` delimiter, generation stops early, since that's the signal it learned for "end of piece."
# 
# - Once the token sequence is finished, `save_melody` walks through it and merges each starting symbol with its trailing `_` continuations into a single note or rest of the right duration. The result is written out to `results/` as a MIDI file you can listen to.
# %%
# Define a wrapper around the trained model for generation and decoding.
class MelodyGenerator:
    """
    A wrapper around the trained LSTM model for generating new melodies.

    This class handles:
    - Loading the trained model from disk
    - Generating new token sequences autoregressively
    - Decoding tokens back into MIDI format
    """

    # Load the trained model and token mapping needed for generation.
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
        # Store the computed value for the next step in the pipeline.
        self.model = keras.models.load_model(str(model_path))

        # Load the token → integer mapping from JSON file
        # Open the target file or resource using a safe context manager.
        with mapping_path.open("r") as fp:
            # Store the annotated value for later use.
            self._mappings: dict[str, int] = json.load(fp)

        # Create reverse mapping: integer → token
        # This is used during generation to convert predicted integers back to strings
        # Store the annotated value for later use.
        self._id_to_token: dict[int, str] = {v: k for k, v in self._mappings.items()}

        # Starting symbols: SEQUENCE_LENGTH "/" tokens
        # This provides a clean context window for generation (no song content yet)
        # Store the annotated value for later use.
        self._start_symbols: list[str] = ["/"] * SEQUENCE_LENGTH

    # Generate a melody continuation one token at a time.
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
        # Keep the provided seed tokens as the initial context.
        seed_tokens = seed

        # Initialize melody with seed tokens (we'll append generated tokens to this)
        # Start the output token list with the seed melody.
        melody = seed_tokens.copy()

        # Build the initial context: start symbols + seed tokens, converted to integers
        # Build the integer context window used for generation.
        context = [self._mappings[t] for t in self._start_symbols + seed_tokens]

        # Generate new tokens one by one
        # Iterate through the current collection.
        for _ in range(num_steps):
            # Get the last max_sequence_length tokens from context
            # This is what we feed to the model (the sliding window)
            # Store the computed value for the next step in the pipeline.
            window = context[-max_sequence_length:]

            # Convert integers to one-hot vectors for model input
            # Store the computed value for the next step in the pipeline.
            onehot = keras.utils.to_categorical(window, num_classes=len(self._mappings))

            # Add batch dimension: (seq_length, vocab_size) → (1, seq_length, vocab_size)
            # Store the computed value for the next step in the pipeline.
            onehot = onehot[np.newaxis, ...]

            # Get model's probability predictions for the next token
            # predictions shape: (1, vocab_size) - probabilities over all tokens
            # Read the model prediction for the next token.
            probabilities = self.model.predict(onehot, verbose=0)[0]

            # Sample next token using temperature scaling
            # Sample the next token id from the predicted distribution.
            next_id = self._sample_with_temperature(probabilities, temperature)

            # Convert the integer ID back to a token string
            # Convert the sampled token id back into a string.
            next_token = self._id_to_token[next_id]

            # Add the sampled integer to context (for future predictions)
            # Append the current item to the growing collection.
            context.append(next_id)

            # If we hit the delimiter token, stop generation
            # Check the current condition before continuing the pipeline.
            if next_token == "/":
                break

            # Add the token to our generated melody
            # Append the current item to the growing collection.
            melody.append(next_token)

        # Return the full list of generated tokens
        # Return the full seed plus generated melody token list.
        return melody


    @staticmethod
    # Sample the next token id from a temperature-scaled distribution.
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
        # Store the computed value for the next step in the pipeline.
        log_probs = np.log(np.maximum(probabilities, 1e-10)) / max(temperature, 1e-6)

        # Subtract the maximum for numerical stability (prevents overflow)
        # This doesn't change the relative values, just shifts everything
        log_probs -= np.max(log_probs)

        # Convert back from log space: exp(log_probs) gives scaled probabilities
        # Store the computed value for the next step in the pipeline.
        dist = np.exp(log_probs)

        # Normalize so probabilities sum to 1
        dist /= dist.sum()

        # Sample one index from the distribution
        # Return the computed result from this helper.
        return int(np.random.choice(len(dist), p=dist))


    # Convert generated tokens back into a MIDI file.
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
        # Create an empty stream for the reconstructed melody.
        stream = m21.stream.Stream()

        # Track the current note/rest being built
        # Store the annotated value for later use.
        start_symbol: Optional[str] = None

        # Count how many time steps (steps) the current note/rest spans
        # Store the annotated value for later use.
        step_counter: int = 1

        # Define a helper for the melody pipeline.
        def _flush() -> None:
            """
            Helper function: finalize the current note/rest and add it to the stream.
            """
            # If we haven't started a note/rest yet, nothing to do
            # Check the current condition before continuing the pipeline.
            if start_symbol is None:
                # Return the computed result from this helper.
                return

            # Calculate total duration: number of steps × step_duration
            # Example: 4 steps × 0.25 = 1.0 quarter length (quarter note)
            # Store the computed value for the next step in the pipeline.
            ql = step_duration * step_counter

            # Create either a Rest or a Note depending on the symbol
            # Check the current condition before continuing the pipeline.
            if start_symbol == "r":
                # "r" means rest
                # Create a note or rest event for the output stream.
                event = m21.note.Rest(quarterLength=ql)
            else:
                # Otherwise it's a MIDI pitch number (convert to int)
                # Create a note or rest event for the output stream.
                event = m21.note.Note(int(start_symbol), quarterLength=ql)

            # Add the note/rest to the stream
            # Append the current item to the growing collection.
            stream.append(event)

        # Process each token in the melody
        # Walk through the generated tokens in order.
        for i, symbol in enumerate(melody):
            # Check if this is a continuation token "_" AND we're not at the end
            # Check the current condition before continuing the pipeline.
            if symbol != "_" or i + 1 == len(melody):
                # This is a new note/rest, so flush the previous one
                # Call the _flush helper to continue the pipeline.
                _flush()

                # Start a new note/rest with this symbol
                # Store the computed value for the next step in the pipeline.
                start_symbol = symbol

                # Reset step counter for the new note/rest
                # Store the computed value for the next step in the pipeline.
                step_counter = 1
            else:
                # This is a continuation token, extend the current note/rest
                step_counter += 1

        # Write the stream to a MIDI file
        # Write the reconstructed melody stream to a MIDI file.
        stream.write("midi", str(file_name))
        # Print a status message so progress stays visible.
        print(f"Melody saved to {file_name}")
# %% [markdown]
# This cell sets up the helpers that pick seed material for generation. `extract_seed_tokens_from_midi` reads a MIDI file and converts the first few seconds of it into the same token format the model was trained on. It walks through the events one by one, adding tokens until it has covered the requested duration. If a single event would run past that limit, the function clips it down so the seed ends exactly where we want.
# 
# The other two helpers just decide where the seed comes from. `choose_seed_random` picks a handful of files at random from the melody directory using a fixed RNG seed, so the same selection comes back every run. `choose_seed_file` is for when you want to point the generator at one specific file instead of leaving it up to chance.
# %%
# Extract the opening seed tokens from a MIDI file.
def extract_seed_tokens_from_midi(midi_path: Path,seed_seconds: float,time_step: float = TIME_STEP,) -> list[str]:
    """Extract seed tokens from the first N seconds of a MIDI file."""
    # Store the computed value for the next step in the pipeline.
    score = m21.converter.parse(str(midi_path))
    # Store the annotated value for later use.
    tokens: list[str] = []

    # Store the computed value for the next step in the pipeline.
    elapsed = 0.0
    # Walk through the flattened score and collect each note or chord.
    for event in score.flatten().notesAndRests:
        # Store the computed value for the next step in the pipeline.
        event_seconds = event.seconds if event.seconds is not None else 0.0
        # Store the computed value for the next step in the pipeline.
        event_seconds = float(event_seconds)

        # Fallback if seconds context is unavailable in parsed stream.
        # Check the current condition before continuing the pipeline.
        if event_seconds <= 0:
            # Store the computed value for the next step in the pipeline.
            event_seconds = float(event.duration.quarterLength)

        # Stop once the requested seed length has been reached.
        if elapsed >= seed_seconds:
            break

        # Store the computed value for the next step in the pipeline.
        symbol = str(int(event.pitch.midi)) if isinstance(event, m21.note.Note) else "r"

        # Keep the whole event if it still fits in the seed window.
        if elapsed + event_seconds <= seed_seconds:
            # Store the computed value for the next step in the pipeline.
            steps = max(1, int(round(float(event.duration.quarterLength) / time_step)))
            # Append the current item to the growing collection.
            tokens.append(symbol)
            # Extend the current collection with additional tokens.
            tokens.extend(["_"] * (steps - 1))
            elapsed += event_seconds
        else:
            # Store the computed value for the next step in the pipeline.
            remaining = max(0.0, seed_seconds - elapsed)
            # Store the computed value for the next step in the pipeline.
            frac = remaining / max(event_seconds, 1e-8)
            # Store the computed value for the next step in the pipeline.
            clipped_q = float(event.duration.quarterLength) * frac
            # Store the computed value for the next step in the pipeline.
            steps = max(1, int(round(clipped_q / time_step)))
            # Append the current item to the growing collection.
            tokens.append(symbol)
            # Extend the current collection with additional tokens.
            tokens.extend(["_"] * (steps - 1))
            break

    # Return the token sequence that was just built.
    return tokens


# Choose a reproducible random subset of MIDI files.
def choose_seed_random(num_files: int, seed: int) -> list[Path]:
    # Seed files are sampled from the same directory used for training data.
    # Store the computed value for the next step in the pipeline.
    candidates = iter_midi_files(MELODIES_DIR)
    # Fail early if there are no MIDI files to choose from.
    if not candidates:
        # Raise a clear error so the notebook fails fast when data is missing.
        raise ValueError(f"No MIDI files found under {MELODIES_DIR}")
    # Store the computed value for the next step in the pipeline.
    rng = np.random.default_rng(seed)
    # Store the computed value for the next step in the pipeline.
    n_selected = min(num_files, len(candidates))
    return list(rng.choice(candidates, size=n_selected, replace=False))

# Validate and return one user-specified seed file.
def choose_seed_file(file_path: Path) -> list[Path]:
    if not file_path.exists():
        raise ValueError(f"Specified seed file not found: {file_path}")
    return [file_path]
# %% [markdown]
# #### Randomly chosen seed files from training set
# 
# - Run the full generation process on a handful of randomly picked melodies from the training set.
# - For each seed file, we extract the opening as a prompt, hand it off to the `MelodyGenerator`, and save the continuation as a new MIDI file.
# - The new MIDI file is saved in `results/` and the filename is appended with the seed file.
# %%
# Store the computed value for the next step in the pipeline.
mg = MelodyGenerator(MODEL_PATH, MAPPING_PATH)
# Pick the MIDI files that will be used as seeds.
selected_seed_files = choose_seed_random(NUM_SEED_FILES, RANDOM_SEED)

# Generate one continuation for each selected seed file.
for seed_midi_path in selected_seed_files:
    # Keep the provided seed tokens as the initial context.
    seed_tokens = extract_seed_tokens_from_midi(
        midi_path=seed_midi_path,
        seed_seconds=SEED_SECONDS,
        time_step=TIME_STEP,
    )

    # Start the output token list with the seed melody.
    melody = mg.generate_melody(
        seed=seed_tokens,
        num_steps=NUM_STEPS,
        max_sequence_length=SEQUENCE_LENGTH,
        temperature=TEMPERATURE,
    )

    # Build the destination path for the generated MIDI file.
    output_path = RESULTS_DIR / f"{seed_midi_path.stem}_generated.mid"
    mg.save_melody(melody, file_name=output_path, step_duration=TIME_STEP)

    print(f"Seed file          : {seed_midi_path.name}")
    print(f"Seed tokens        : {len(seed_tokens)}")
    print(f"Generated tokens   : {len(melody)}")
    print(f"Output MIDI        : {output_path}")
# %% [markdown]
# #### User-specified seed file
# 
# - Same generation pipeline as above, but pointed at a specific MIDI file instead of a random one from the training set.
# 
# <span style="color:red"> Note: Due to limited training time and dataset, not all notes or even durations from an arbitrary MIDI files will be recognized by the model. This often results in a crash </span>
# %%

# Store the computed value for the next step in the pipeline.
mg = MelodyGenerator(MODEL_PATH, MAPPING_PATH)
# Pick the MIDI files that will be used as seeds.
selected_seed_files = choose_seed_file(Path("happy_birthday.mid"))

# Generate one continuation for each selected seed file.
for seed_midi_path in selected_seed_files:
    # Keep the provided seed tokens as the initial context.
    seed_tokens = extract_seed_tokens_from_midi(
        midi_path=seed_midi_path,
        seed_seconds=SEED_SECONDS,
        time_step=TIME_STEP,
    )

    # Start the output token list with the seed melody.
    melody = mg.generate_melody(
        seed=seed_tokens,
        num_steps=NUM_STEPS,
        max_sequence_length=SEQUENCE_LENGTH,
        temperature=TEMPERATURE,
    )

    # Build the destination path for the generated MIDI file.
    output_path = RESULTS_DIR / f"{seed_midi_path.stem}_generated.midi"
    mg.save_melody(melody, file_name=output_path, step_duration=TIME_STEP)

    print(f"Seed file          : {seed_midi_path.name}")
    print(f"Seed tokens        : {len(seed_tokens)}")
    print(f"Generated tokens   : {len(melody)}")
    print(f"Output MIDI        : {output_path}")
