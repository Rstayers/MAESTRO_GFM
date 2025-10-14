from typing import List, Dict


class UCBPlanner:
    def __init__(self, capability, k: int = 1, kappa: float = 0.0, diversity: bool = True):
        self.cap = capability
        self.k = k
        self.kappa = kappa
        self.diversity = diversity


    def select(self, phi_vec, tools_meta: List[Dict]):
        scores = []
        for t in tools_meta:
            mu, sd = self.cap.predict(t['name'], phi_vec)
            scores.append((t, mu + self.kappa * sd))
        scores.sort(key=lambda x: x[1], reverse=True)
        chosen = []
        used = set()
        for t, _ in scores:
            fam = t.get('family', 'NA')
            if not self.diversity or fam not in used:
                chosen.append(t)
                used.add(fam)
            if len(chosen) >= self.k: break
        return chosen