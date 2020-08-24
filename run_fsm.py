import subprocess
from multiprocessing import Pool

if __name__=='__main__':
    experiments = [
        {'--env_name': 'predator_prey',
         '--device': '0',
         '--nagents': '3',
         '--hid_size': '128',
         '--detach_gap': '10',
         '--dim': '5',
         '--max_steps': '20',
         '--ic3net': None,
         '--vision': '0',
         '--recurrent': None,
         '--mode': 'cooperative',
         '--comm_action_one': None,
         '--qbn': None,
         '--load': 'logs/predator-prey/dim=5-n=3/cooperative/model'
         },
        {'--env_name': 'predator_prey',
         '--device': '1',
         '--nagents': '3',
         '--hid_size': '128',
         '--detach_gap': '10',
         '--dim': '5',
         '--max_steps': '20',
         '--ic3net': None,
         '--vision': '0',
         '--recurrent': None,
         '--mode': 'mixed',
         '--comm_action_one': None,
         '--qbn': None,
         '--load': 'logs/predator-prey/dim=5-n=3/mixed/model'
         },
        {'--env_name': 'predator_prey',
         '--device': '2',
         '--nagents': '3',
         '--hid_size': '128',
         '--detach_gap': '10',
         '--dim': '5',
         '--max_steps': '20',
         '--ic3net': None,
         '--vision': '0',
         '--recurrent': None,
         '--mode': 'competitive',
         '--comm_action_one': None,
         '--qbn': None,
         '--load': 'logs/predator-prey/dim=5-n=3/competitive/model'
         },
        {'--env_name': 'traffic_junction',
         '--device': '3',
         '--nagents': '5',
         '--hid_size': '128',
         '--detach_gap': '10',
         '--dim': '6',
         '--max_steps': '20',
         '--ic3net': None,
         '--vision': '0',
         '--recurrent': None,
         '--add_rate_min': '0.1',
         '--add_rate_max': '0.3',
         '--curr_start': '250',
         '--curr_end': '1250',
         '--difficulty': 'easy',
         '--comm_action_one': None,
         '--qbn': None,
         '--load': 'logs/traffic_junction/easy/model'
         },
        {'--env_name': 'traffic_junction',
         '--device': '4',
         '--nagents': '10',
         '--hid_size': '128',
         '--detach_gap': '10',
         '--dim': '14',
         '--max_steps': '40',
         '--ic3net': None,
         '--vision': '0',
         '--recurrent': None,
         '--add_rate_min': '0.05',
         '--add_rate_max': '0.2',
         '--curr_start': '250',
         '--curr_end': '1250',
         '--difficulty': 'medium',
         '--comm_action_one': None,
         '--qbn': None,
         '--load': 'logs/traffic_junction/medium/model'
         },
        {'--env_name': 'traffic_junction',
         '--device': '5',
         '--nagents': '20',
         '--hid_size': '128',
         '--detach_gap': '10',
         '--dim': '18',
         '--max_steps': '80',
         '--ic3net': None,
         '--vision': '0',
         '--recurrent': None,
         '--add_rate_min': '0.02',
         '--add_rate_max': '0.05',
         '--curr_start': '250',
         '--curr_end': '1250',
         '--difficulty': 'hard',
         '--comm_action_one': None,
         '--qbn': None,
         '--load': 'logs/traffic_junction/hard/model'
         },
    ]
    def run_experiment(experiment):
        cmd = ['python', 'learn_fsm.py']
        print(experiment)
        for key, value in experiment.items():
            cmd.append(key)
            if value is not None:
                cmd.append(value)
        return subprocess.call(cmd)

    pool = Pool(6)
    pool.map(run_experiment, experiments)
    pool.close()
