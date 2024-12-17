import torch
import torch.nn.functional as F

def phase_unwrap(phase):
    """
    Perform phase unwrapping to ensure continuity across frames.
    
    Args:
        phase (Tensor): Phase tensor of shape (B, #frames, #freq_bins).
    
    Returns:
        Tensor: Unwrapped phase tensor.
    """
    # Compute phase difference between consecutive frames
    phase_diff = torch.diff(phase, dim=1)
    
    # Identify jumps larger than pi
    phase_diff_mod = torch.remainder(phase_diff + torch.pi, 2 * torch.pi) - torch.pi
    
    # Cumulative sum to reconstruct unwrapped phase
    phase_unwrapped = torch.cumsum(torch.cat([phase[:, :1], phase_diff_mod], dim=1), dim=1)
    
    return phase_unwrapped

def compute_stft_frame_lengths(lengths, fft_size, shift_size, win_length):
    """
    Compute the number of frames after STFT for given signal lengths.
    
    Args:
        lengths (Tensor): Original lengths of the signals (B,).
        fft_size (int): FFT size.
        shift_size (int): Hop size.
        win_length (int): Window length.
    
    Returns:
        Tensor: Frame lengths after STFT (B,).
    """
    # Ensure lengths are greater than window size
    effective_lengths = (lengths - win_length).clamp(min=0)
    
    # Compute number of frames
    frame_lengths = (effective_lengths // shift_size) + 1
    
    return frame_lengths

def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)
    return x_stft

class PhaseLoss(torch.nn.Module):
    """Phase loss module."""

    def __init__(self):
        """Initialize phase loss module."""
        super(PhaseLoss, self).__init__()

    def forward(self, x_stft, y_stft, lengths):
        """Calculate forward propagation.
        Args:
            x_stft (Tensor): STFT of predicted signal (B, #frames, #freq_bins).
            y_stft (Tensor): STFT of groundtruth signal (B, #frames, #freq_bins).
            lengths (Tensor): Valid Frame lengths of batch (B,).
        Returns:
            Tensor: Phase loss value.
        """
        # Unwrap phases
        x_phase = phase_unwrap(torch.angle(x_stft))
        y_phase = phase_unwrap(torch.angle(y_stft))

        # Compute phase difference
        phase_diff = torch.remainder(x_phase - y_phase + torch.pi, 2 * torch.pi) - torch.pi

        # Apply mask based on frame lengths
        max_frames = x_phase.size(1)
        mask = torch.arange(max_frames, device=x_phase.device).expand(len(lengths), max_frames) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1)
        phase_diff = phase_diff * mask

        # Compute normalized loss
        phase_loss = torch.sum(torch.abs(phase_diff)) / torch.sum(mask)

        return phase_loss

class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag, lengths):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module considering variable lengths."""
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag, lengths):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
            lengths (Tensor): Valid flame lengths
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        total_loss = F.l1_loss(torch.log(y_mag), torch.log(x_mag), reduction="sum")
        return total_loss / torch.sum(lengths)
      
class CrossSpectralLoss(torch.nn.Module):
    """Cross-spectral loss module."""

    def __init__(self):
        """Initialize cross-spectral loss module."""
        super(CrossSpectralLoss, self).__init__()

    def forward(self, x_stft, y_stft, lengths):
        """Calculate forward propagation.
        Args:
            x_stft (Tensor): STFT of predicted signal (B, #frames, #freq_bins).
            y_stft (Tensor): STFT of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Cross-spectral loss value.
        """
        cross_spectrum = x_stft * y_stft.conj()
        mag_diff = torch.abs(torch.abs(cross_spectrum) - torch.abs(y_stft * y_stft.conj()))
        return torch.sum(mag_diff)/torch.sum(lengths)


class STFTLoss(torch.nn.Module):
    """STFT loss module with cross-spectral loss."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.spectral_convergence_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
        #self.cross_spectral_loss = CrossSpectralLoss()
        self.phase_loss = PhaseLoss()

    def forward(self, x, y, lengths):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
            lengths (Tensor): Lengths of each sample in the batch (B,).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
            Tensor: Cross-spectral loss value.
        """
        x_stft = stft(x.cuda(), self.fft_size, self.shift_size, self.win_length, self.window)
        y_stft = stft(y.cuda(), self.fft_size, self.shift_size, self.win_length, self.window)
        
        # Create masks based on lengths
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1)  # Adjust dimensions for STFT output
        
        x_stft = x_stft * mask
        y_stft = y_stft * mask
        
        x_mag = torch.abs(x_stft)
        y_mag = torch.abs(y_stft)

        valid_lengths = compute_stft_frame_lengths(lengths, self.fft_size, self.shift_size, self.win_length)
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag, valid_lengths)
        #cross_loss = self.cross_spectral_loss(x_stft, y_stft, valid_lengths)
        ph_loss = self.phase_loss(x_stft, y_stft, valid_lengths)
      
        return sc_loss, mag_loss, ph_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module with cross-spectral loss, considering variable lengths."""

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window", factor_sc=0.1, factor_mag=0.1, factor_phase=0.1):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            factor_sc (float): Balancing factor for spectral convergence loss.
            factor_mag (float): Balancing factor for magnitude loss.
            factor_phase (float): Balancing factor for phase-spectral loss.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag
        self.factor_phase = factor_phase

    def forward(self, x, y, lengths):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
            lengths (Tensor): Lengths of each sample in the batch (B,).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
            Tensor: Multi resolution cross-spectral loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        phase_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l, cross_l = f(x, y, lengths)
            sc_loss += sc_l
            mag_loss += mag_l
            cross_loss += cross_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)
        phase_loss /= len(self.stft_losses)

        return self.factor_sc * sc_loss, self.factor_mag * mag_loss, self.factor_phase * phase_loss