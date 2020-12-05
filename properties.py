from argparse import ArgumentParser
import logging

##LOGGING PROPERTY
LOG_FILE = 'logfile'
CONSOLE_LEVEL = logging.INFO
LOGFILE_LEVEL = logging.DEBUG

def build_parser():
    parser = ArgumentParser()

    parser.add_argument("--mode", dest="mode", metavar="mode", default="train")
    parser.add_argument("--refer_img", dest="refer_img", metavar="refer_img", default=None)


    parser.add_argument("--device", dest="device", metavar="device", default="gpu")
    parser.add_argument("--env",dest="env", metavar="env", default="BreakoutDeterministic-v4")
    parser.add_argument("--memory_size", dest="memory_size", metavar="memory_size", type=int, default=int(1e6))
    parser.add_argument("--update_freq", dest="update_freq", metavar="update_freq", type=int, default=4)
    parser.add_argument("--learn_start", dest="learn_start", metavar="learn_start", type=int, default=50000)
    parser.add_argument("--history_size", dest="history_size", metavar="history_size", type=int, default=4)
    parser.add_argument("--target_update", dest="target_update", metavar="target_update", type=int, default=10000)

    parser.add_argument("--n_ensemble", dest="n_ensemble", type=int, default=9)
    parser.add_argument("--bernoulli_prob", dest="bernoulli_prob", type=float, default=0.9)

    ##Learning rate
    parser.add_argument("--batch_size", dest="batch_size", metavar="batch_size", type=int, default=32)
    parser.add_argument("--ep", dest="ep", metavar="ep", type=int, default=1)
    parser.add_argument("--eps_end", dest="eps_end", metavar="eps_end", type=float, default=0.01)
    parser.add_argument("--eps_endt", dest="eps_endt", metavar="eps_endt", type=int, default=int(1e6))
    parser.add_argument("--lr", dest="lr", metavar="lr", type=float, default=0.00025)
    parser.add_argument("--discount", dest="discount", metavar="discount", type=float, default=0.99)


    parser.add_argument("--agent_type", dest="agent_type", metavar="agent_type", default="DQN")
    parser.add_argument("--max_steps", dest="max_steps", metavar="max_steps", type=int, default=int(5e6))
    parser.add_argument("--start_steps", dest="start_steps", metavar="start_steps", type=int, default=0)

    parser.add_argument("--eval_freq", dest="eval_freq", metavar="eval_freq", type=int, default=50000)
    parser.add_argument("--eval_steps", dest="eval_steps", metavar="eval_steps", type=int, default=50000)

    parser.add_argument("--max_eval_iter", dest="max_eval_iter", metavar="max_eval_iter", type=int, default=10000)

    parser.add_argument("--pretrained_dir", dest="pretrained_dir", metavar="pretrained_dir", type=str, default=None)
    parser.add_argument("--out_dir", dest="out_dir", metavar="out_dir", type=str, default=None, required=True)



    return parser