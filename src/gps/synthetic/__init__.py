"""Phase-0 synthetic players with *known* dynamic mechanisms.

Before touching real data we construct players whose generating mechanism is
known, so the central claims become falsifiable independent of messy real
data (proposal section 5, Phase 0):

* ``TiltPlayer``        -- plays cleanly until a loss, then blunders more for
                           N games (post-loss tilt).
* ``TimePressurePlayer``-- move quality degrades below T seconds remaining.
* ``FatiguePlayer``     -- quality degrades after game K within a session.

Each is a perturbation of an abstract "engine" policy over a toy game, so
the experiments below are checkable on CPU:

    P0.1 Does the dynamic latent recover the injected mechanism?
    P0.2 Does a static individual model provably fail where dynamics matter?
    P0.3 Does persona prompting over/under-shoot the injected trait?
    P0.4 Calibration and identifiability of z_t under known ground truth.

Pure stdlib + a small seeded RNG so runs are reproducible without numpy.
"""

from gps.synthetic.players import (
    FatiguePlayer,
    SyntheticPlayer,
    TiltPlayer,
    TimePressurePlayer,
)
from gps.synthetic.toy_game import ToyGame, ToyPosition

__all__ = [
    "FatiguePlayer",
    "SyntheticPlayer",
    "TiltPlayer",
    "TimePressurePlayer",
    "ToyGame",
    "ToyPosition",
]
