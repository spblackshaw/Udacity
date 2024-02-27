

class BayesRuleGenerator:
    def __init__(self,
                 prob_yes: float):
        # Prior probability
        self.prior_prob_yes = prob_yes
        self.prior_prob_no = 1 - self.prior_prob_yes
        prior_prob_yes_pos = 0.9
        prior_prob_yes_neg = round(1 - prior_prob_yes_pos, 2)
        prior_prob_no_neg = 0.9
        prior_prob_no_pos = round(1 - prior_prob_no_neg, 2)

        # Posterior
        joint_posterior_yes_pos = self.prior_prob_yes * prior_prob_yes_pos
        joint_posterior_no_pos = self.prior_prob_no * prior_prob_no_pos

        normaliser = joint_posterior_yes_pos + joint_posterior_no_pos
        posterior_yes = joint_posterior_yes_pos / normaliser
        posterior_no = joint_posterior_no_pos / normaliser
        posterior = posterior_yes + posterior_no


BayesRuleGenerator(prob_yes=0.5)
print("here")

