from TALENT.model.methods.base import Method
import torch
from tqdm import tqdm

class TabTransformerMethod(Method):
    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)
        assert(args.cat_policy == 'indices')

    def construct_model(self, model_config = None):
        if model_config is None:
            model_config = self.args.config['model']
        from TALENT.model.models.tabtransformer import TabTransformerModel
        self.model = TabTransformerModel(
            categories=self.categories,
            num_continuous=self.d_in,
            dim_out=self.d_out,
            mlp_act=torch.nn.ReLU(),
            mlp_hidden_mults=(4, 2),
            **model_config
        ).to(self.args.device)
        if self.args.use_float:
            self.model.float()
        else:
            self.model.double()

    def predict_proba(self, data):
        """
        Predict the results of the data.

        :param data: tuple, (N, C, y)
        :param info: dict, information about the data
        :param model_name: str, name of the model
        :return: tuple, (loss, metric, metric_name, predictions)
        """
        N,C = data

        ## Evaluation Stage
        self.model.eval()

        self.data_format(False, N, C)

        test_logit = []
        with torch.no_grad():
            for i, (X, y) in tqdm(enumerate(self.test_loader)):
                if self.N is not None and self.C is not None:
                    X_num, X_cat = X[0], X[1]
                elif self.C is not None and self.N is None:
                    X_num, X_cat = None, X
                else:
                    X_num, X_cat = X, None  
                        
                pred = self.model(X_num, X_cat)

                test_logit.append(pred)
                
        test_logit = torch.cat(test_logit, 0)
        return test_logit