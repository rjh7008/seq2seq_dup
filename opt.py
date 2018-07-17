import argparse



def model_opts(parser):
  parser.add_argument('-batch_size', default=16, type=int,
                      help="batch size")

  parser.add_argument('-hidden_size', default=256, type=int,
                      help="rnn hidden size")

  parser.add_argument('-embed_size', type=int, default=200,
                      help="word embedding size")

  parser.add_argument('-max_vocab', default=40000, type=int,
                      help="max vocab size")

  parser.add_argument('-mode', default='train', type=str,
                      help="Mode selection")

  parser.add_argument('-layers', default=1, type=int,
                      help="number of rnn layer")

