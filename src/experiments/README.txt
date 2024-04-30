Files:

*** Navigate Environment Files ***
ppo_bc_navigate.zip - Full BC+PPO model for navigate task
student_ppo_policy_navigate - Pretrained policy weights using BC for navigate task
student_ppo_navigate - Model from which pretrained policy was taken, SB3 needs this when loading policy

navigate_bc.py - Change file paths in config and DATA global var at beginning of file accordingly to run the following:

	def main():

    		# Trains BC policy using supervised learning
    		#train_bc()

    		# rl_bc = True if you want to continue training BC policy with RL
    		# rl_bc = False if you want just RL
    		#train_rl(rl_bc=True)

    		# bc_only = True if you want to test just the BC policy
    		# bc_only = False if you want to test an RL model
    		test(bc_only=False)

*** Treechop Environment Files ***
ppo_bc_treechop.zip - Full BC+PPO model for treechop task
student_ppo_policy_treechop - Pretrained policy weights using BC for treechop task
student_ppo_treechop - Model from which pretrained policy was taken, SB3 needs this when loading policy

navigate_bc.py - Change file paths in config and DATA global var at beginning of file accordingly to run the following:

	def main():

    		# Trains BC policy using supervised learning
    		#train_bc()

    		# rl_bc = True if you want to continue training BC policy with RL
    		# rl_bc = False if you want just RL
    		#train_rl(rl_bc=True)

    		# bc_only = True if you want to test just the BC policy
    		# bc_only = False if you want to test an RL model
    		test(bc_only=False)
