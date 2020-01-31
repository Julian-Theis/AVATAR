import argparse
import os

from avatar.relgan.real.real_gan.real_train import real_train
from avatar.relgan.real.real_gan.real_loader import RealDataLoader
from avatar.relgan.utils.text_process import text_precess
from avatar.relgan.utils.utils import pp
import avatar.relgan.models as models

def main(given_args=None):
    parser = argparse.ArgumentParser(description='Train and run a RelGAN')
    # Architecture
    parser.add_argument('--gf-dim', default=64, type=int, help='Number of filters to use for generator')
    parser.add_argument('--df-dim', default=64, type=int, help='Number of filters to use for discriminator')
    parser.add_argument('--g-architecture', default='rmc_att', type=str, help='Architecture for generator')
    parser.add_argument('--d-architecture', default='rmc_att', type=str, help='Architecture for discriminator')
    parser.add_argument('--gan-type', default='standard', type=str, help='Which type of GAN to use')
    parser.add_argument('--hidden-dim', default=32, type=int,
                        help='only used for OrcaleLstm and lstm_vanilla (generator)')
    parser.add_argument('--sn', default=False, action='store_true', help='if using spectral norm')

    # Training
    parser.add_argument('--gsteps', default='1', type=int, help='How many training steps to use for generator')
    parser.add_argument('--dsteps', default='5', type=int, help='How many training steps to use for discriminator')
    parser.add_argument('--npre-epochs', default=150, type=int, help='Number of steps to run pre-training')
    parser.add_argument('--nadv-steps', default=5000, type=int, help='Number of steps to run adversarial training')
    parser.add_argument('--ntest', default=50, type=int, help='How often to run tests')
    parser.add_argument('--d-lr', default=1e-4, type=float, help='Learning rate for the discriminator')
    parser.add_argument('--gpre-lr', default=1e-2, type=float, help='Learning rate for the generator in pre-training')
    parser.add_argument('--gadv-lr', default=1e-4, type=float, help='Learning rate for the generator in adv-training')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size for training')
    parser.add_argument('--log-dir', default='./oracle/logs', type=str, help='Where to store log and checkpoint files')
    parser.add_argument('--sample-dir', default='./oracle/samples', type=str,
                        help='Where to put samples during training')
    parser.add_argument('--optimizer', default='adam', type=str, help='training method')
    parser.add_argument('--decay', default=False, action='store_true', help='if decaying learning rate')
    parser.add_argument('--adapt', default='exp', type=str,
                        help='temperature control policy: [no, lin, exp, log, sigmoid, quad, sqrt]')
    parser.add_argument('--seed', default=123, type=int, help='for reproducing the results')
    parser.add_argument('--temperature', default=1000, type=float, help='the largest temperature')

    # evaluation
    parser.add_argument('--nll-oracle', default=False, action='store_true', help='if using nll-oracle metric')
    parser.add_argument('--nll-gen', default=False, action='store_true', help='if using nll-gen metric')
    parser.add_argument('--bleu', default=False, action='store_true', help='if using bleu metric, [2,3,4,5]')
    parser.add_argument('--selfbleu', default=False, action='store_true', help='if using selfbleu metric, [2,3,4,5]')
    parser.add_argument('--doc-embsim', default=False, action='store_true', help='if using DocEmbSim metric')

    # relational memory
    parser.add_argument('--mem-slots', default=1, type=int, help="memory size")
    parser.add_argument('--head-size', default=512, type=int, help="head size or memory size")
    parser.add_argument('--num-heads', default=2, type=int, help="number of heads")

    # Data
    parser.add_argument('--dataset', default='oracle', type=str, help='[oracle, image_coco, emnlp_news]')
    parser.add_argument('--vocab-size', default=5000, type=int, help="vocabulary size")
    parser.add_argument('--start-token', default=0, type=int, help="start token for a sentence")
    parser.add_argument('--seq-len', default=20, type=int, help="sequence length: [20, 40]")
    parser.add_argument('--num-sentences', default=10000, type=int, help="number of total sentences")
    parser.add_argument('--gen-emb-dim', default=32, type=int, help="generator embedding dimension")
    parser.add_argument('--dis-emb-dim', default=64, type=int, help="TOTAL discriminator embedding dimension")
    parser.add_argument('--num-rep', default=64, type=int, help="number of discriminator embedded representations")
    parser.add_argument('--data-dir', default='/data/julian/data/relgan/data', type=str,
                        help='Where data data is stored')

    if given_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args=given_args)
    pp.pprint(vars(args))
    config = vars(args)

    print(config)

    data_file = os.path.join(args.data_dir, '{}.txt'.format(args.dataset))

    if args.dataset == 'pb_system_4_1_10':
        args.batch_size = 32

    seq_len, vocab_size = text_precess(data_file)
    config['seq_len'] = seq_len
    config['vocab_size'] = vocab_size
    print('seq_len: %d, vocab_size: %d' % (seq_len, vocab_size))

    oracle_loader = RealDataLoader(args.batch_size, args.seq_len)

    generator = models.get_generator(args.g_architecture, vocab_size=vocab_size, batch_size=args.batch_size,
                                         seq_len=seq_len, gen_emb_dim=args.gen_emb_dim, mem_slots=args.mem_slots,
                                         head_size=args.head_size, num_heads=args.num_heads, hidden_dim=args.hidden_dim,
                                         start_token=args.start_token)
    discriminator = models.get_discriminator(args.d_architecture, batch_size=args.batch_size, seq_len=seq_len,
                                                 vocab_size=vocab_size, dis_emb_dim=args.dis_emb_dim,
                                                 num_rep=args.num_rep, sn=args.sn)

    prob_discriminator = models.get_prob_discriminator(args.d_architecture, batch_size=args.batch_size, seq_len=seq_len,
                                                 vocab_size=vocab_size, dis_emb_dim=args.dis_emb_dim,
                                                 num_rep=args.num_rep, sn=args.sn)

    real_train(generator, discriminator, oracle_loader, config, prob_discriminator=prob_discriminator)

if __name__ == '__main__':
    main()
