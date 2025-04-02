


def main(args, resume_preempt=False):
    '''
    Pseudo code
    args.get("data name")
    #1 initial parameter and build env
    for loop until train steps done 
        ## sample and train
        #2 sample part
            if replay.size < warmup_length:
                action = random_policy(obs)
            else:
                emb_code = world_model(obs)
                action = agent(emb_code)
            obs = env.step(action)
            obs,info update
            env done handling
        #3 training world model 
            if replay.size >= warmup_length:
                clip(video, action...) = replay.sample()
                world_model.update(clip)
        #4 training agent
            if replay.size >= warmup_length:
                imagined_data = utils.imagine_with_world_model(agent, world_model)
                agent.update(imagined_data)
        #5 logging
            if step % log_interval == 0:
                logger.log_metrics()
            
            if step % save_interval == 0:
                save_model(world_model, agent)
    
    '''
    
    
    pass