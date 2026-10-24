import random
from collections import defaultdict

class MinimalNISSGERModel:
    """
    Minimal public implementation of NISSGER:
    A non-inferential ensemble recorder that forms state sequences
    purely from observed transitions.
    """

    def __init__(self):
        # Estimated transition frequencies
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        # Internal record of observed states
        self.internal_history = []
        # External environment history (for inspection only)
        self.environment_history = []

    def observe(self, state):
        """Record an observed state and update transition frequencies."""
        self.environment_history.append(state)

        if self.internal_history:
            prev = self.internal_history[-1]
            self.transition_matrix[prev][state] += 1

        self.internal_history.append(state)

    def simulate_chain(self, start_state, length=10):
        """Generate a sequence by repeatedly selecting the most frequent next state."""
        seq = [start_state]
        current = start_state

        for _ in range(length - 1):
            next_candidates = self.transition_matrix[current]
            if not next_candidates:
                break
            next_state = max(next_candidates, key=next_candidates.get)
            seq.append(next_state)
            current = next_state

        return seq

    def predict_next(self, state):
        """Predict the next state based on observed transition frequencies."""
        next_candidates = self.transition_matrix[state]
        if not next_candidates:
            return None
        return max(next_candidates, key=next_candidates.get)


# ------------------------------------------------------------
# Context-dependent environment generator (hidden context)
# ------------------------------------------------------------

def generate_contextual_environment(length=50):
    """
    Generate an environment where hidden contexts alter transition tendencies.
    The model cannot observe the context; it only records the resulting sequence.
    """
    contexts = ["A", "B"]
    env_states = ["cold", "cool", "warm", "hot"]

    transition_rules = {
        "A": {
            "cold": ["cool", "warm"],
            "cool": ["warm"],
            "warm": ["hot"],
            "hot": ["cool"]
        },
        "B": {
            "cold": ["hot"],
            "cool": ["cold"],
            "warm": ["cool"],
            "hot": ["warm"]
        }
    }

    environment = []
    current_state = random.choice(env_states)
    current_context = random.choice(contexts)

    for _ in range(length):
        environment.append(current_state)

        next_candidates = transition_rules[current_context][current_state]
        current_state = random.choice(next_candidates)

        # Occasionally switch context
        if random.random() < 0.1:
            current_context = random.choice(contexts)

    return environment


# ------------------------------------------------------------
# Sequence comparison utilities
# ------------------------------------------------------------

def compute_sequence_accuracy(model, environment):
    """Compare replayed sequence with the environment sequence."""
    if not environment:
        return 0, 0, []

    start_state = environment[0]
    replay_seq = model.simulate_chain(start_state, length=len(environment))

    matches = sum(
        1 for i in range(1, len(environment))
        if i < len(replay_seq) and replay_seq[i] == environment[i]
    )

    return matches, len(environment) - 1, replay_seq


def print_parallel_sequences(environment, replay_seq):
    """Print environment and replay sequences side-by-side."""
    print("\n=== Environment vs Replay ===")
    print("index | environment | replay")
    print("--------------------------------")
    for i in range(min(len(environment), len(replay_seq))):
        print(f"{i:5d} | {environment[i]:11s} | {replay_seq[i]}")


# ------------------------------------------------------------
# Example execution
# ------------------------------------------------------------

if __name__ == "__main__":
    environment = generate_contextual_environment(50)
    model = MinimalNISSGERModel()

    for state in environment:
        model.observe(state)

    print("=== Environment ===")
    print(model.environment_history)

    print("\n=== Estimated Transition Matrix ===")
    for s in model.transition_matrix:
        print(s, "→", dict(model.transition_matrix[s]))

    matches, total, replay_seq = compute_sequence_accuracy(model, environment)

    print("\n=== Replay Sequence ===")
    print(replay_seq)

    print(f"\n=== Accuracy (start = first environment state) ===\n{matches}/{total} matches")

    print_parallel_sequences(environment, replay_seq)

    test_state = environment[20]
    print(f"\n=== Next-State Prediction (state = {test_state}) ===")
    print("Predicted next state:", model.predict_next(test_state))
