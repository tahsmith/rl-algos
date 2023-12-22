def q_learning(policy, gamma, alpha):
   def episode_fn(state, step, i, Q):
       done = False
       while not done:
           action = policy(Q, i, state)
           next_state, reward, done, info = step(action)
           td_err = (reward + gamma * Q[next_state, :].max() * (not done)
                     - Q[state, action])
           Q[state, action] += alpha(i) * td_err
           state = next_state
       return Q

   return episode_fn
