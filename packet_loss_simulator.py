import numpy as np

def _compute_transition_probs(
    plr: float, lamda: float, p_good: float, p_bad: float
) -> tuple[float, float]:
    """
    Compute transition probabilities for a Gilbert-Elliot model.

    Returns:
        p_good_to_bad: Probability of transitioning from Good to Bad state.
        p_bad_to_good: Probability of transitioning from Bad to Good state.
    """
    plr = max(plr, 0.0)
    ratio = (p_bad - plr) / max(p_bad - p_good, 1e-8)
    p_alpha = (1.0 - lamda) * (1.0 - ratio)
    p_beta = (1.0 - lamda) * ratio
    return p_alpha, p_beta


def simulate_gilbert_elliot(
    length: int,
    plr: float = 0.1,
    lamda: float = 0.5,
    p_good: float = 0.0,
    p_bad: float = 0.5,
    start_good: bool = True,
) -> np.ndarray:
    """
    Simulate a Gilbert-Elliot packet loss sequence.

    Args:
        length: Number of packets to simulate.
        plr: Target long-term packet loss rate. If plr <= 0.0, no loss occurs.
        lamda: Burstiness parameter (0-1).
        p_good: Loss rate in Good state.
        p_bad: Loss rate in Bad state.
        start_good: Whether to start in Good state.

    Returns:
        seq: Array of 1 (received) / 0 (lost), dtype int8.
    """
    if plr <= 0.0:
        return np.ones(length, dtype=np.int8)

    p_gb, p_bg = _compute_transition_probs(plr, lamda, p_good, p_bad)
    seq = np.empty(length, dtype=np.int8)
    current_state = 1 if start_good else 0
    seq[0] = current_state

    for i in range(1, length):
        r = np.random.rand()
        if current_state == 1:
            if r < p_gb:
                current_state = 0
        else:
            if r < p_bg:
                current_state = 1
        seq[i] = current_state

    return seq


class GilbertElliotModel:
    """
    Gilbert-Elliot packet loss simulator wrapper.

    Example:
        model = GilbertElliotModel(plr=0.1, lamda=0.5)
        seq = model.simulate(100)
    """
    def __init__(
        self,
        plr: float = 0.1,
        lamda: float = 0.5,
        p_good: float = 0.0,
        p_bad: float = 0.5,
    ):
        self.plr = max(plr, 0.0)
        self.lamda = lamda
        self.p_good = p_good
        self.p_bad = p_bad
        self.p_gb, self.p_bg = _compute_transition_probs(
            self.plr, self.lamda, self.p_good, self.p_bad
        )

    def simulate(
        self,
        length: int,
        start_good: bool = True,
    ) -> np.ndarray:
        """
        Simulate packet reception (1) and loss (0).

        Args:
            length: Number of packets.
            start_good: Start in Good state if True.

        Returns:
            seq: np.ndarray of shape (length,), dtype int8.
        """
        if self.plr <= 0.0:
            return np.ones(length, dtype=np.int8)

        return simulate_gilbert_elliot(
            length,
            plr=self.plr,
            lamda=self.lamda,
            p_good=self.p_good,
            p_bad=self.p_bad,
            start_good=start_good,
        )


if __name__ == '__main__':
    # No loss example
    model_no_loss = GilbertElliotModel(plr=0.0, lamda=0.5)
    print("All received:", model_no_loss.simulate(10))

    # Normal loss example
    model = GilbertElliotModel(plr=0.1, lamda=0.5)
    print("Sample loss:", model.simulate(10))
