#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extractor classes for splitting a MIDI piano roll into melody, harmony, rhythm, and dynamics components"""

from copy import deepcopy
from itertools import groupby
from operator import itemgetter
from typing import Optional, Any

import numpy as np
from numba import jit
from pretty_midi import PrettyMIDI, Instrument, Note

from .utils import PIANO_KEYS, CLIP_LENGTH, HOP_SIZE, FPS, MIDI_OFFSET, MAX_VELOCITY

MINIMUM_FRAMES = 1  # a note must last this number of frames to be used
QUANTIZE_RESOLUTION = (1 / FPS) * 10  # 100 ms, the duration of a triplet eighth note at the mean JTD tempo (200 BPM)

__all__ = [
    "normalize_array", "get_piano_roll", "quantize", "augment_midi", "MelodyExtractor", "DynamicsExtractor",
    "RhythmExtractor", "HarmonyExtractor", "ExtractorError", "RollExtractor", "note_list_to_midi"
]


class ExtractorError(Exception):
    """Custom extractor error for catching bad clips in individual piano roll extractor classes"""
    pass


@jit(nopython=True)
def normalize_array(track_array: np.ndarray) -> np.ndarray:
    """Normalize values in a 2D array between 0 and 1"""
    return (track_array - np.min(track_array)) / (np.max(track_array) - np.min(track_array))


@jit(nopython=True, fastmath=True)
def get_piano_roll(clip_notes) -> np.ndarray:
    """Largely the same as PrettyMIDI.get_piano_roll, but with optimizations using Numba"""
    clip_array = np.zeros((PIANO_KEYS, CLIP_LENGTH * FPS), dtype=np.float32)
    frame_iter = np.linspace(0, CLIP_LENGTH, (CLIP_LENGTH * FPS) + 1)
    for frame_idx, (frame_start, frame_end) in enumerate(zip(frame_iter, frame_iter[1:])):
        for note in clip_notes:
            note_start, note_end, note_pitch, note_velocity = note
            if note_start < frame_end and note_end >= frame_start:
                clip_array[note_pitch - MIDI_OFFSET, frame_idx] = note_velocity
    return clip_array


def quantize(midi_time: float, resolution: float = QUANTIZE_RESOLUTION) -> float:
    """Snap a given `midi_time` value to the nearest `resolution` interval"""
    return np.round(midi_time / resolution) * resolution


def validate_midi(input_midi: Any) -> PrettyMIDI:
    """Checks input PrettyMIDI object and raises ExtractorError when problems are found"""
    # Input must be a PrettyMIDI object
    if not isinstance(input_midi, PrettyMIDI):
        raise ExtractorError(f"expected PrettyMIDI input, but found {type(input_midi)}")
    # Input must contain instruments
    if len(input_midi.instruments) < 1:
        raise ExtractorError("no instruments found in PrettyMIDI input; does the input contain any MIDI events?")
    # First instrument must contain notes
    if len(input_midi.instruments[0].notes) < 1:
        raise ExtractorError("MIDI input does not contain any notes")
    return input_midi


def note_list_to_midi(midi_notes: list[tuple], resolution: int = 384, program: int = 0) -> PrettyMIDI:
    """Converts a list of tuples in format (start, end, pitch, velocity) to a PrettyMIDI file"""
    # Create empty PrettyMIDI and Instrument classes using the same settings as our input
    pm_obj = PrettyMIDI(resolution=resolution)
    instr = Instrument(program=program)
    # Iterate through each note and convert them to PrettyMIDI format
    for note in midi_notes:
        note_start, note_end, note_pitch, note_velocity = note
        instr.notes.append(Note(start=note_start, end=note_end, pitch=note_pitch, velocity=note_velocity))
    # Set the notes to the instrument object, and the instrument object to the PrettyMIDI object
    pm_obj.instruments.append(instr)
    return pm_obj


def augment_midi(midi_obj: PrettyMIDI, **kwargs) -> PrettyMIDI:
    """Applies data augmentation to a MIDI object, consisting of random transposition, time dilation, and rebalancing"""
    input_midi = validate_midi(midi_obj)
    # Specific parameters for each effect
    transpose_limit = kwargs.get("transpose_limit", 6)
    dilate_limit = kwargs.get("dilate_limit", 0.2)
    balance_limit = kwargs.get("balance_limit", 12)
    # Calculate the amount to transpose and dilate by
    transpose_amount = np.random.randint(-transpose_limit, transpose_limit)
    dilate_amount = np.random.uniform(1 - dilate_limit, 1 + dilate_limit)
    # Iterate through all the notes in the MIDI object
    augmented_notes = []
    for note in input_midi.instruments[0].notes:
        # Unpack the pretty MIDI note object
        note_start, note_end, note_pitch, note_velocity = note.start, note.end, note.pitch, note.velocity
        # Add the constant to the pitch value to transpose
        note_pitch += transpose_amount
        # Multiply the start and end time by the constant to dilate it
        note_start *= dilate_amount
        note_end *= dilate_amount
        # Add a RANDOM value to each note velocity, within the limit
        note_velocity += np.random.randint(-balance_limit, balance_limit)
        # Clip note velocity between 0 and 127
        note_velocity = max(0, min(note_velocity, MAX_VELOCITY))
        augmented_notes.append((note_start, note_end, note_pitch, note_velocity))
    # Return the output in PrettyMIDI format again with the same metadata as our input
    output = note_list_to_midi(augmented_notes, input_midi.resolution, input_midi.instruments[0].program)
    return output


class RollExtractor:
    """Base extractor class from which all others inherit. Can be used to extract a single channel piano roll."""

    def __init__(self, midi_obj: PrettyMIDI, clip_start: float = 0.0, **kwargs):
        self.input_midi = validate_midi(midi_obj)
        # Define start time for the clip
        self.clip_start = clip_start
        self.clip_end = self.clip_start + CLIP_LENGTH
        # Convert notes to a tuple for easier working
        raw = [(note.start, note.end, note.pitch, note.velocity) for note in self.input_midi.instruments[0].notes]
        # Truncate notes to start and end of clip times and apply validation checks
        prepared = list(self.preparation(raw))
        # Transform data into correct format: this should be different, depending on the Extractor class
        transformed = list(self.transformation(prepared))
        # Create output in both PrettyMIDI and piano roll format
        self.roll = self.create_roll(transformed)  # Will raise an ExtractorError if no notes present
        self.output_midi = note_list_to_midi(
            transformed, self.input_midi.resolution, self.input_midi.instruments[0].program
        )

    def validate_note(self, midi_note: tuple[float, float, int, int]) -> bool:
        """Returns True if a note fulfils all required conditions for the extractor, False if not"""
        note_start, note_end, note_pitch, note_velocity = midi_note
        return all((
            # Note must fit within the piano keyboard
            MIDI_OFFSET <= note_pitch < PIANO_KEYS + MIDI_OFFSET,
            # Note must start within the clip
            self.clip_start <= note_start <= self.clip_end,  # We can keep notes that extend past the clip
            # Note must last for at least N frames (default = 1)
            (note_end - note_start) > (MINIMUM_FRAMES / FPS),
            # Note must have a velocity within the acceptable MIDI range
            0 < note_velocity < MAX_VELOCITY,
        ))

    def preparation(self, midi_notes: list[tuple]) -> list[tuple]:
        """Prepare the MIDI object by truncating to the (possibly random) clip boundaries and validating each note"""
        # Iterate through all notes, sorted by start time
        for note in sorted(midi_notes, key=itemgetter(0)):
            note_start, note_end, note_pitch, note_velocity = note
            # Check the note: e.g., does it exist within the clip boundaries we'll use for our roll?
            if self.validate_note((note_start, note_end, note_pitch, note_velocity)):
                # Express the start and end of notes with relation to the start of the clip
                note_start -= self.clip_start
                note_end -= self.clip_start
                yield note_start, note_end, note_pitch, note_velocity

    def transformation(self, validated_midi_notes: list[tuple]) -> list[tuple]:
        """Apply transformations to individual MIDI notes. Should be updated in a child class."""
        for note in validated_midi_notes:
            yield note

    @staticmethod
    def create_roll(transformed_midi_notes: list[tuple]):
        """Try and create a piano roll array, raises an `ExtractorError` when no notes are present"""
        try:
            return get_piano_roll(transformed_midi_notes)
        # We could simply use the non-jit function with get_piano_roll.py_func
        # But I think it makes more sense to raise the error, catch the resulting None in the dataloader + collate_fn,
        # and remove this bad clip from the batch so we don't use it during training.
        except ValueError:
            raise ExtractorError('no notes present in clip when creating piano roll!')

    def heatmap(self, ax: Optional = None):
        """Creates a heatmap of the piano roll in seaborn, returns a matplotlib Axes object for use in a Figure"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Create a new axis object, if none is passed
        if ax is None:
            ax = plt.gca()
        # Create the heatmap in seaborn
        sns.heatmap(normalize_array(self.roll), ax=ax, mask=self.roll == 0, cmap='mako', cbar=None)
        # Set some plot aesthetics
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth(1)
        ax.grid()
        # Set axes limits correctly
        ax.set(
            xticks=range(0, 3500, 500), yticks=range(0, PIANO_KEYS, 12),
            xticklabels=range(0, 3500, 500), yticklabels=range(0, PIANO_KEYS, 12)
        )
        # Return the axes object to be incorporated in a larger plot
        return ax

    def synthesize(self, output_path: str, delete_midi: bool = True) -> None:
        """Renders the output MIDI from the extractor as a .wav file using fluidsynth"""
        import os
        import subprocess

        # Render the transformed MIDI as a file
        self.output_midi.write(output_path + '.mid')
        # Call Fluidsynth in subprocess to render the MIDI
        cmd = f'fluidsynth -ni -r 44100 -F "{output_path}.wav" "{output_path}.mid"'
        subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        # Remove the MIDI file if required
        if delete_midi:
            os.remove(output_path + '.mid')


class MelodyExtractor(RollExtractor):
    """Extracts melody using skyline algorithm, while removing rhythm, harmony, and velocity"""

    def __init__(self, midi_obj: PrettyMIDI, clip_start: float = 0, **kwargs):
        # Get parameter settings from kwargs
        # TODO: we should check the interval between successive notes as well
        # self.lower_bound = kwargs.get('lower_bound', 40)  # By default, use all notes above the E below middle C
        self.lower_bound = kwargs.get('lower_bound', MIDI_OFFSET)  # Use all notes by default
        self.quantize_resolution = kwargs.get('quantize_resolution', 1 / FPS)  # Snap to nearest 10 ms
        # Special attribute that holds the results of skylining without adjusting duration
        self.skylined = None
        self.accomp = None
        # Initialise the parent class and call all the overridden methods
        super().__init__(midi_obj, clip_start=clip_start)

    def transformation(self, validated_midi_notes: list[tuple]):
        """Transform the MIDI by quantizing onset/offset times, binarizing velocity, and applying skyline."""
        quantized_notes = []
        # Iterate through all notes
        for note in validated_midi_notes:
            note_start, note_end, note_pitch, note_velocity = note
            # If the note is above our lower bound
            if note_pitch >= self.lower_bound:
                # Snap note start and end with the given time resolution
                note_start = quantize(note_start, self.quantize_resolution)
                note_end = quantize(note_end, self.quantize_resolution)
                # Set the velocity to maximum to binarize
                note_velocity = MAX_VELOCITY  # 1 = note on, 0 = note off
                quantized_notes.append((note_start, note_end, note_pitch, note_velocity))
        # Apply the skyline algorithm and adjust the duration of notes
        melody, accomp = self.apply_skyline(quantized_notes)
        # Set these special attributes which contains original note durations with the skyline applied/not applied
        #  we do this to make our life easier when creating the web application
        self.skylined = note_list_to_midi(melody, self.input_midi.resolution, self.input_midi.instruments[0].program)
        self.accomp = note_list_to_midi(accomp, self.input_midi.resolution, self.input_midi.instruments[0].program)
        # Return the skyline melody with note durations adjusted
        return self.adjust_durations(melody)

    @staticmethod
    def apply_skyline(quantized_notes: list):
        """For all notes with the same (quantized) start time, keep only the highest pitch note"""
        note_starts = itemgetter(0)
        # Group all the MIDI events by the (quantized) start time
        grouper = groupby(sorted(quantized_notes, key=note_starts), note_starts)
        melody, accompaniment = [], []
        # Iterate through each group
        for quantized_note_start, subiter in grouper:
            subiter = list(subiter)  # we need to refer to the list several times
            # Get the note with the highest pitch in the group and append to our melody
            max_pitch = max(subiter, key=itemgetter(2))
            melody.append(max_pitch)
            # Extend the accompaniment list with every other note
            accompaniment.extend([i for i in subiter if i[2] != max_pitch[2]])
        return melody, accompaniment

    @staticmethod
    def adjust_durations(skylined_notes: list):
        """For a list of notes N and piece with duration T, adjust duration of notes n = T / len(N)"""
        try:
            note_duration = CLIP_LENGTH / len(skylined_notes)
        # This error will be caught in the dataloader collate_fn and the bad clip will be skipped.
        # This logic here allows us to note which clips are broken.
        except ZeroDivisionError:
            raise ExtractorError('no notes present in clip when extracting melody.')
        else:
            for note_idx, note in enumerate(skylined_notes):
                # Keep the pitch and (randomized) velocity of each note
                _, __, note_pitch, note_velocity = note
                # Adjust the note start and end time so that the duration of each note is the same
                note_start = note_idx * note_duration
                note_end = (note_idx + 1) * note_duration
                if note_idx == len(skylined_notes):
                    note_end = CLIP_LENGTH
                yield note_start, note_end, note_pitch, note_velocity


class HarmonyExtractor(RollExtractor):
    """Extracts harmony by grouping together near-simultaneous notes, and removes rhythm, dynamics, and melody"""

    def __init__(self, midi_obj: PrettyMIDI, clip_start: float = 0, **kwargs):
        # Get the parameter settings from the kwargs, with defaults
        # self.upper_bound = kwargs.get('upper_bound', 69)  # Use all notes below A above middle C by default
        self.upper_bound = kwargs.get('upper_bound', PIANO_KEYS + MIDI_OFFSET)  # Use all notes by default
        self.minimum_notes = kwargs.get('minimum_notes', 3)  # Only consider triads or greater
        self.quantize_resolution = kwargs.get('quantize_resolution', 0.3)  # Snap to 300ms, i.e. 1/4 note at mean JTD
        self.remove_highest_pitch = kwargs.get("remove_highest_pitch", False)
        self.chorded = None
        # Initialise the parent class and call all the overridden methods
        super().__init__(midi_obj, clip_start=clip_start)

    def transformation(self, validated_midi_notes: list[tuple]) -> list[tuple]:
        """Quantize note start and end, randomize velocity, and keep pitch"""
        quantized_notes = []
        # Iterate through all notes
        for note in validated_midi_notes:
            note_start, note_end, note_pitch, note_velocity = note
            # If the note is above our lower bound
            if note_pitch <= self.upper_bound:
                note_start, note_end, note_pitch, note_velocity = note
                # Snap note start and end with the given time resolution
                note_start = quantize(note_start, self.quantize_resolution)
                note_end = quantize(note_end, self.quantize_resolution)
                quantized_notes.append((note_start, note_end, note_pitch, note_velocity))
        # Group notes into chords and adjust the durations
        self.chorded = list(self.group_chords(quantized_notes))
        return self.adjust_durations(self.chorded)

    def group_chords(self, notes: list):
        """Group simultaneous notes with at least MINIMUM_NOTES into chords (iterables of notes)"""
        note_starts = itemgetter(0)
        # Group all the MIDI events by the (quantized) start time
        grouper = groupby(sorted(notes, key=note_starts), note_starts)
        # Iterate through each group and keep only the note with the highest pitch
        for quantized_note_start, subiter in grouper:
            # Get the unique pitches in the chord
            chord = list(subiter)
            pitches = set(c[2] for c in chord)
            # Remove the top note from each chord, if we've specified this option
            if self.remove_highest_pitch:
                highest_pitch = max(pitches)
                pitches = {i for i in pitches if i != highest_pitch}
                chord = [c for c in chord if c[2] != highest_pitch]
            # Keep the chord if it has at least MINIMUM_NOTES unique pitches (i.e., not counting duplicates)
            if all((
                    len(pitches) >= self.minimum_notes,
                    # max(pitches) - min(pitches) <= self.MAX_SPAN
            )):
                yield chord

    @staticmethod
    def adjust_durations(chords: list):
        """For a list of notes N and piece with duration T, adjust duration of notes n = T / len(N)"""
        try:
            chord_duration = CLIP_LENGTH / len(chords)
        # This error will be caught in the dataloader collate_fn and the bad clip will be skipped.
        # This logic here allows us to note which clips are broken.
        except ZeroDivisionError:
            raise ExtractorError('no notes present in clip when extracting harmony.')
        else:
            # Iterate through every chord (this is an iterable of notes)
            for note_idx, chord in enumerate(chords):
                # Get the adjusted note start and end time
                chord_start = note_idx * chord_duration
                chord_end = (note_idx + 1) * chord_duration
                # Set the velocity to binary
                chord_velocity = MAX_VELOCITY  # 1 = note on, 0 = note off
                if chord_end > CLIP_LENGTH:
                    chord_end = CLIP_LENGTH
                # Iterate through each note in the chord and set the times correctly, keeping pitch + velocity
                for note in chord:
                    _, __, note_pitch, ___ = note
                    yield float(chord_start), float(chord_end), note_pitch, chord_velocity


class RhythmExtractor(RollExtractor):
    """Extracts rhythm (note onset and offset times), while shuffling pitch and removing velocity"""

    def __init__(self, midi_obj: PrettyMIDI, clip_start: float = 0, **kwargs):
        # Initialise the parent class and call all the overridden methods
        super().__init__(midi_obj, clip_start=clip_start)

    # No need to overwrite validation logic, RollExtractor code works fine

    def transformation(self, validated_midi_notes: list[tuple]) -> list[tuple]:
        for note in validated_midi_notes:
            note_start, note_end, note_pitch, note_velocity = note
            # Randomise the pitch
            note_pitch = np.random.randint(MIDI_OFFSET, (PIANO_KEYS + MIDI_OFFSET) - 1)
            # Set the velocity to binary
            note_velocity = MAX_VELOCITY  # 1 = note on, 0 = note off
            yield note_start, note_end, note_pitch, note_velocity


class DynamicsExtractor(RollExtractor):
    """Extracts velocity while randomising pitch and removing note onset and offset times"""

    def __init__(self, midi_obj: PrettyMIDI, clip_start: float = 0, **kwargs):
        # Get parameter settings from kwargs
        self.quantize_resolution = kwargs.get('quantize_resolution', 1 / FPS)  # Snap to nearest 10 ms
        # Initialise the parent class and call all the overridden methods
        super().__init__(midi_obj, clip_start=clip_start)

    def transformation(self, validated_midi_notes: list[tuple]) -> list[tuple]:
        """Adjust note durations and randomize pitch"""
        quantized_notes = []
        # Iterate notes
        for note in validated_midi_notes:
            note_start, note_end, note_pitch, note_velocity = note
            # Snap note start and end with the given time resolution
            note_start = quantize(note_start, self.quantize_resolution)
            note_end = quantize(note_end, self.quantize_resolution)
            # Randomise the pitch
            note_pitch = np.random.randint(MIDI_OFFSET, (PIANO_KEYS + MIDI_OFFSET) - 1)
            quantized_notes.append((note_start, note_end, note_pitch, note_velocity))
        # Adjust duration of all quantized notes
        return self.adjust_durations(quantized_notes)

    @staticmethod
    def adjust_durations(notes: list[tuple]) -> list[tuple]:
        note_starts = itemgetter(0)
        # Group by the note start time and get the duration of each quantized bin
        grouper = groupby(sorted(notes, key=note_starts), note_starts)
        try:
            note_duration = CLIP_LENGTH / len(list(deepcopy(grouper)))
        # This error will be caught in the dataloader collate_fn and the bad clip will be skipped.
        # This logic here allows us to note which clips are broken.
        except ZeroDivisionError:
            raise ExtractorError('no notes present in clip when extracting velocity.')
        else:
            # Iterate through the notes in each quantized bin
            for note_idx, (quantized_note_start, subiter) in enumerate(grouper):
                # Set the start and end time for this bin
                note_start = note_idx * note_duration
                note_end = (note_idx + 1) * note_duration
                # Iterate through each individual note in the bin
                for note in subiter:
                    _, __, note_pitch, note_velocity = note
                    yield note_start, note_end, note_pitch, note_velocity


def create_outputs_from_extractors(track_name: str, extractors: list) -> None:
    """Creates .WAV and .PNG files for a particular track with extractors and raw_midi"""
    import matplotlib.pyplot as plt

    # Create the folder for this track, in our reports directory
    ref_folder = os.path.join(get_project_root(), 'reports/midi_test', track_name)
    if not os.path.exists(ref_folder):
        os.makedirs(ref_folder)
    # Create the stacked piano roll object
    extractors_names = ["Raw", "Melody", "Harmony", "Rhythm", "Dynamics"]
    fig, ax = plt.subplots(5, 1, figsize=(10, 10), sharex=True, sharey=True)
    for a, ext, nme in zip(ax.flatten(), extractors, extractors_names):
        if ext is not None:
            a = ext.heatmap(a)
            a.set(title=nme)
            # Create the synthesized .WAV file, while we're iterating
            ext.synthesize(os.path.join(ref_folder, nme.lower()), delete_midi=False)
    # Plot aesthetics
    ax[0].invert_yaxis()
    fig.supylabel('MIDI Note')
    fig.supxlabel('Frame')
    fig.tight_layout()
    # Save inside our reports directory
    fig.savefig(os.path.join(ref_folder, 'heatmap.png'))


if __name__ == "__main__":
    import os
    from time import time
    from tqdm import tqdm
    from .utils import get_project_root, seed_everything
    from loguru import logger

    seed_everything(10)
    # Get the tracks we'll create clips from
    n_clips = 1000
    use_in_test = np.random.choice(os.listdir(os.path.join(get_project_root(), 'data/clips/pijama')), n_clips)
    # We'll store results in here
    res = {"base": [], "melody": [], "harmony": [], "rhythm": [], "dynamics": []}
    # A simple iterable of extractor classes
    all_extractors = [RollExtractor, MelodyExtractor, HarmonyExtractor, RhythmExtractor, DynamicsExtractor]
    # Probability that we'll augment our MIDI
    augment_prob = 1.0  # i.e., always augment

    # Iterate through each track
    for track_idx, test_track in tqdm(enumerate(use_in_test), desc=f'Making {n_clips} clips...', total=n_clips):
        # Create the filepath and skip if it does not exist (e.g., `.gitkeep` files)
        test_fpath = os.path.join(get_project_root(), 'data/clips/pijama', test_track, 'clip_000.mid')
        if not os.path.isfile(test_fpath):
            continue
        # Load in the MIDI object
        test_midi_obj = PrettyMIDI(test_fpath)
        extracted_results = []
        # Augment the MIDI object and randomize the start time of the clip
        if np.random.uniform(0.0, 1.0) <= augment_prob:
            test_midi_obj = augment_midi(test_midi_obj)
        # Randomize the start time for each clip between 0 and 15 seconds
        cs = np.random.uniform(0.0, HOP_SIZE // 2)
        # Create each extractor
        for name, extractor in zip(res.keys(), all_extractors):
            start = time()
            try:
                created = extractor(test_midi_obj, clip_start=cs)
            # Log a warning if we failed to create the extractor
            except ExtractorError as err:
                logger.warning(f'Failed to create {name} for {test_track}, skipping! {err}')
                extracted_results.append(None)
            else:
                extracted_results.append(created)
            finally:
                res[name].append(time() - start)
        # For every 10 items, create a graph and WAV files
        if track_idx % 10 == 0:
            create_outputs_from_extractors(test_track, extracted_results)
    # Create the results string and log
    fmt_res = ", ".join(
        f"{k}: total {np.mean(v):.2f}s, mean {np.mean(v):.2f}s, SD {np.std(v):.2f}s" for k, v in res.items()
    )
    logger.info(f"For {n_clips} clips, results: {fmt_res}")
