def sample_trajectories_train(env, N, m, l, g, build_dics_func):
    dicts_in_static = []
    dicts_in_dynamic = []
    dicts_out_static = []
    dicts_out_dynamic = []
    state = env.reset()
    while (len(dicts_in_static) < N):
      for i in range(100):

          action = env.action_space.sample()
          next_state, _, done, _ = env.step(action)

          dict_in_static, dict_in_dynamic = build_dics_func(m, l, g, state, action)
          dict_out_static, dict_out_dynamic = build_dics_func(m, l, g, next_state, action)
          for key in ["nodes","globals"]:
              dict_out_static[key] -= dict_in_static[key]
              dict_out_dynamic[key] -= dict_in_dynamic[key]
          dicts_in_static.append( dict_in_static )
          dicts_in_dynamic.append( dict_in_dynamic )
          dicts_out_static.append( dict_out_static )
          dicts_out_dynamic.append( dict_out_dynamic )

          state = next_state

          if done or len(dicts_in_static) == N:
              break
    return dicts_in_static, dicts_in_dynamic, dicts_out_static, dicts_out_dynamic

def sample_trajectories_test(env, N, m, l, g, build_dics_func):
    dicts_in_static = []
    dicts_in_dynamic = []
    dicts_out_static = []
    dicts_out_dynamic = []
    state = env.reset()
    while (len(dicts_in_static) < N):
      for i in range(N):

          action = env.action_space.sample()
          next_state, _, done, _ = env.step(action)

          dict_in_static, dict_in_dynamic = build_dics_func(m, l, g, state, action)
          dict_out_static, dict_out_dynamic = build_dics_func(m, l, g, next_state, action)
          dicts_in_static.append( dict_in_static )
          dicts_in_dynamic.append( dict_in_dynamic )
          dicts_out_static.append( dict_out_static )
          dicts_out_dynamic.append( dict_out_dynamic )

          state = next_state

          if done or len(dicts_in_static) == N:
              return dicts_in_static, dicts_in_dynamic, dicts_out_static, dicts_out_dynamic
    return dicts_in_static, dicts_in_dynamic, dicts_out_static, dicts_out_dynamic

