from utils.config import parse_args
from utils.data_loader import get_data_loader
from models.dpp_model import DPP
from utils.visdom_graph import Visdom

def main(args):
    model = None
    graph = Visdom(args)
    
    if args.model =='vgg16':
        model = DPP(args)
    elif args.model == 'dpp_vgg16':
        model = DPP(args)

    train_loader, test_loader = get_data_loader(args)
    if args.is_train == 'True':
        model.train(train_loader, test_loader, graph)
    else:
        acc = model.test(test_loader, 0, graph)
        print("test acc is", acc)
if __name__ == '__main__':
    args = parse_args()
    main(args)
