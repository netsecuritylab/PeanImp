import mainUtils
import train_evalCopy

def main():
    args = mainUtils.setup_main_params()
    args.embway = 'pretrain'
    config = mainUtils.get_config(args)
    train_data = mainUtils.prepare_data(config, args)
    
    valid_loss, valid_acc, valid_fpr, valid_tpr, valid_ftf, valid_f1 = 0, 0, 0, 0, 0, 0
    #model = mainUtils.get_model(config)
    # TODO: maybe don't instantiate the model before the loop?
    for i in range(config.k):
        train_, test_ = mainUtils.get_k_fold_data(config.k, i, train_data)
        model = mainUtils.get_model(config)

        train_iter = mainUtils.build_iterator(train_, config)
        test_iter = mainUtils.build_iterator(test_, config)
        dev_iter = mainUtils.build_iterator(test_, config)
        acc_, loss_, f1_, fpr_, tpr_, ftf_ = train_evalCopy.train(config, model, train_iter, dev_iter, test_iter)

        print('*' * 25, 'result of', i + 1, 'fold', '*' * 25)
        print('loss:%.6f' % loss_, 'acc:%.4f' % acc_, 'FPR:%.4f' % fpr_, 'TPR:%.4f' % tpr_, 'FTF:%.4f' % ftf_, 'F1-macro:%.4f' % f1_, "\n")
        valid_loss += loss_
        valid_acc += acc_
        valid_fpr += fpr_
        valid_tpr += tpr_
        valid_ftf += ftf_
        valid_f1 += f1_
    print("\n", '#' * 10, 'final result of all k fold', '#' * 10)
    print('acc:%.4f' % (valid_acc/config.k), 'F1-macro:%.4f' % (valid_f1/config.k), \
          'TPR:%.4f' % (valid_tpr/config.k), 'FPR:%.4f' % (valid_fpr/config.k), \
          'FTF:%.4f' % (valid_ftf/config.k), 'loss:%.6f' % (valid_loss/config.k), "\n")


if __name__ == '__main__':
    main()