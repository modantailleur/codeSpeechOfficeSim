"""A-weighted level computation and a simple room acoustics simulator."""
import numpy as np
from scipy.signal import bilinear, lfilter
import pyroomacoustics as pra


def a_weighting(fs):
    """
    Design digital A-weighting filter (approximate)
    Returns b, a for lfilter
    """
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997

    # Analog coefficients
    NUMs = [(2*np.pi*f4)**2*(10**(A1000/20)), 0, 0, 0, 0]
    DENs = np.polymul([1 + 2*np.pi*f4, (2*np.pi*f4)**2],
                      np.polymul([1 + 2*np.pi*f1, 2*np.pi*f1],
                                 np.polymul([1 + 2*np.pi*f3, 2*np.pi*f3],
                                            [1 + 2*np.pi*f2, 2*np.pi*f2])))

    b, a = bilinear(NUMs, DENs, fs)
    return b, a


def compute_LAeq(signal, fs):
    # Convert to mono if stereo
    if signal.ndim > 1:
        signal = signal.mean(axis=0)
    # Apply A-weighting
    b, a = a_weighting(fs)
    signal_a = lfilter(b, a, signal)
    # RMS over time
    rms = np.sqrt(np.mean(signal_a**2))
    # Convert to dBA
    LAeq = 20 * np.log10(rms + 1e-12)
    return LAeq


class RoomAcousticProcessor:
    """
    Process audio signals for room acoustic analysis.
    """
    def __init__(self, sr, distance=2.0, room_dim=[10, 10, 3], mic_spacing=0.20):
        """
        Initialize with sample rate.

        Args:
            fs: Sample rate in Hz
        """
        self.distance = distance
        # --------------------------------------------------
        # Room
        # --------------------------------------------------
        self.room = pra.ShoeBox(
            room_dim,
            fs=sr,
            materials=pra.Material(0.80),
            max_order=5,
            air_absorption=True
        )

        # --------------------------------------------------
        # Stereo microphone array
        # --------------------------------------------------
        self.mic_center = np.array([room_dim[0]/2, room_dim[1]/2, 1.5])

        self.mic_positions = np.array([
            self.mic_center + [-mic_spacing/2, 0, 0],
            self.mic_center + [ mic_spacing/2, 0, 0]
        ]).T  # (3, 2)

        self.room.add_microphone_array(
            pra.MicrophoneArray(self.mic_positions, sr)
        )

    def process(self, signal, pan=0.0):
        """Compute A-weighted equivalent level."""
        # --------------------------------------------------
        # Source position (pan → azimuth)
        # --------------------------------------------------
        self.room.sources = []
        azimuth = pan * (np.pi / 2)   # ±90°

        src_pos = self.mic_center + np.array([
            self.distance * np.sin(azimuth),
           -self.distance * np.cos(azimuth),
            0
        ])

        self.room.add_source(src_pos, signal=signal)
        self.room.simulate()
        out = self.room.mic_array.signals  # (samples, 2)

        return out
