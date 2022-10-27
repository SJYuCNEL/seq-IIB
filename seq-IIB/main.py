import argparse
from train import dataset_selection, model_selection, setup_seed, train, try_gpu

def get_parser():
    parser = argparse.ArgumentParser(description='GIB IRM')

    #experiment parameters
    parser.add_argument('--seed',type=int,default=520,help='random seed of model MNIST:520')
    parser.add_argument('--as_one',type=bool,default=False,help='Whether to set the sequential environment scene, default value: False')
    parser.add_argument('--experiment_time',type=str,default='_4',help='Experiment number')
    parser.add_argument('--ways_to_train',type=str,default='seq-IIB',help="ERM,IRM,IRMG,Gate,IIB,IBIRM,seq-IIB")

    #this model
    parser.add_argument('--type_datasets',type=str,default='MNIST',help="type_datasets ['MNIST','FashionMNIST','KMNIST','EMNIST']")#Except for MNIST other datasets are only available in 4 environments
    parser.add_argument('--domain_number',type=int,default=4,help='domain number [2 ,4 ,8 ] , Number of environments')
    parser.add_argument('--classes_number',type=int,default=2,help='Number of classes')
    parser.add_argument('--num_epochs',type=int,default=400,help='Number of epochs [2:400 , 4:400 , 8:400]')
    parser.add_argument('--batch_size',type=int,default=256,help='batch size MNIST:256')
    parser.add_argument('--lr',type=float,default=2.5e-4,help='MNIST:2.5e-4')
    parser.add_argument('--weight_decay',type=float,default=0.00125,help='weight_decay')

    parser.add_argument('--length',type=int,default=28)
    parser.add_argument('--width',type=int,default=28)
    parser.add_argument('--height',type=int,default=3)
    parser.add_argument('--hidden_dim',type=int,default=200)
    parser.add_argument('--num_classes',type=int,default=2)
    parser.add_argument('--dropp',type=float,default=0)
    parser.add_argument('--mask_type',type=str,default='tanh',help='mask_type')

    #GIB & g_GIB
    parser.add_argument('--net_freeze_epoch',type=int,default=100,help='net_freeze_epoch [2:200 , 4:100 , 8:50], Freeze the network')
    parser.add_argument('--up_mask_epoch',type=int,default=1,help='up_mask_epoch MNIST:1  Frequency of updating the mask')
    parser.add_argument('--up_mask_epoch_after',type=int,default=1,help='up_mask_epoch MNIST:1  Frequency of updating the mask after freezing the network')
    parser.add_argument('--lambda0before',type=float,default=1.,help='lambda0 MNIST:1')
    parser.add_argument('--lambda1before',type=float,default=1e-4,help='lambda1 MNIST:1e-4')
    parser.add_argument('--lambda2before',type=float,default=1,help='lambda2 MNIST:1')
    parser.add_argument('--pbefore',type=float,default=1.,help='p MNIST:1')
    parser.add_argument('--lambda0after',type=float,default=1.,help='lambda0 MNIST:1')
    # parser.add_argument('--lambda1after',type=float,default=1e-3,help='lambda1')
    parser.add_argument('--lambda1after',type=float,default=6,help='lambda1 MNIST:3 fashionMNIST:5 KMNIST:3')
    parser.add_argument('--lambda2after',type=float,default=10,help='lambda2 MNIST:10')
    parser.add_argument('--pafter',type=float,default=1,help='p MNIST:1')
    parser.add_argument('--feature_number_rate',type=float,default=0.5,help='P MNIST:0.5')
    parser.add_argument('--feature_number_rate_after',type=float,default=0.5,help='P MNIST:0.5')
    parser.add_argument('--tau',type=float,default=1,help='gumbel softmax MNIST:1')

    #IRM
    parser.add_argument('--irm_loss',type=float,default=0.5)

    #IIB
    parser.add_argument('--lambda_beta',type=float,default=1e-4)
    parser.add_argument('--lambda_inv_risks',type=float,default=10)

    #IBIRM
    parser.add_argument('--penalty_weight',type=float,default=0.1)
    parser.add_argument('--ib_penalty_weight',type=float,default=10)

    #Interpretability
    parser.add_argument('--Inter',type=bool,default=True,help='Visualization experiments')
    parser.add_argument('--gray',type=bool,default=True,help='Contrast the accuracy of the test with gray images')
    parser.add_argument('--imgs_number',type=int,default=16,help='Visualize part of the data')
    parser.add_argument('--img_idx',type=int,default=2,help='Select data for SC map')#0,2,4,5,9

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    setup_seed(10)#0,10,20,30,520  Set random seeds for the dataset
    train_data,test_data = dataset_selection(args)

    setup_seed(args.seed)#Set random seeds for the model
    model = model_selection(args)
    
    train(args,model,train_data,test_data,try_gpu())