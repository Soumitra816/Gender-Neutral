import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    ByteTensor = torch.cuda.ByteTensor

else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    ByteTensor = torch.ByteTensor


class MaskedNLLLoss_hu(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss_hu, self).__init__()
        self.weight = weight
        self.loss = nn.BCEWithLogitsLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        
        pred=pred
        mask_ = mask.reshape(-1,1) # batch*seq_len, 1
        if type(self.weight)==type(None):
           
            loss = self.loss(pred*mask_, target.view(-1, 1))/torch.sum(mask)
        else:
            
            loss = self.loss(pred*mask_, target.view(-1, 1))\
                            /torch.sum(self.weight[target]*mask_.squeeze())
        return loss
class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.BCEWithLogitsLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred1,pred2, target1,target2, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        
        pred1=pred1
        pred2=pred2
        mask_ = mask.reshape(-1,1) # batch*seq_len, 1
        if type(self.weight)==type(None):
           
            loss1 = self.loss(pred1*mask_, target1.view(-1, 3))/torch.sum(mask)
            loss2 = self.loss(pred2*mask_, target2.view(-1, 7))/torch.sum(mask)
            #print("i m not the right one")
        else:
            
            loss1 = self.loss(pred1*mask_, target1.view(-1, 1))\
                            /torch.sum(self.weight[target]*mask_.squeeze())
            loss2 = self.loss(pred2*mask_, target2.view(-1, 1))\
                            /torch.sum(self.weight[target]*mask_.squeeze())
        loss=loss1+loss2
        return loss

class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len
        target -> batch*seq_len
        mask -> batch*seq_len
        """
        loss = self.loss(pred*mask, target)/torch.sum(mask)
        return loss


class UnMaskedWeightedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(UnMaskedWeightedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        """
        if type(self.weight)==type(None):
            loss = self.loss(pred, target)
        else:
            loss = self.loss(pred, target)\
                            /torch.sum(self.weight[target])
        return loss


class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim,1,bias=False)

    def forward(self, M, x=None):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M) # seq_len, batch, 1
        alpha = F.softmax(scale, dim=0).permute(1,2,0) # batch, 1, seq_len
        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # batch, vector
        return attn_pool, alpha


class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type!='concat' or alpha_dim!=None
        assert att_type!='dot' or mem_dim==cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type=='general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type=='general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
            #torch.nn.init.normal_(self.transform.weight,std=0.01)
        elif att_type=='concat':
            self.transform = nn.Linear(cand_dim+mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim)
        mask -> (batch, seq_len)
        """
        if type(mask)==type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type=='dot':
            # vector = cand_dim = mem_dim
            M_ = M.permute(1,2,0) # batch, vector, seqlen
            x_ = x.unsqueeze(1) # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general2':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2) # batch, seq_len, mem_dim
            M_ = M_ * mask_
            alpha_ = torch.bmm(x_, M_)*mask.unsqueeze(1)
            alpha_ = torch.tanh(alpha_)
            alpha_ = F.softmax(alpha_, dim=2)
            # alpha_ = F.softmax((torch.bmm(x_, M_))*mask.unsqueeze(1), dim=2) # batch, 1, seqlen
            alpha_masked = alpha_*mask.unsqueeze(1) # batch, 1, seqlen
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True) # batch, 1, 1
            alpha = alpha_masked/alpha_sum # batch, 1, 1 ; normalized
            #import ipdb;ipdb.set_trace()
        else:
            M_ = M.transpose(0,1) # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1,M.size()[0],-1) # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_,x_],2) # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_)) # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a),1).transpose(1,2) # batch, 1, seqlen

        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # batch, mem_dim
        return attn_pool, alpha


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
            
        score = F.softmax(score, dim=0)
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score
    
class CNNFeatureExtractor(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, output_size, filters, kernel_sizes, dropout):
        super(CNNFeatureExtractor, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim, out_channels=filters, kernel_size=K) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * filters, output_size)
        self.feature_dim = output_size


    def init_pretrained_embeddings_from_numpy(self, pretrained_word_vectors):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        # if is_static:
        self.embedding.weight.requires_grad = False


    def forward(self, x, umask):
        
        num_utt, batch, num_words = x.size()
        
        x = x.type(LongTensor)  # (num_utt, batch, num_words)
        x = x.view(-1, num_words) # (num_utt, batch, num_words) -> (num_utt * batch, num_words)
        emb = self.embedding(x) # (num_utt * batch, num_words) -> (num_utt * batch, num_words, embedding_dim) 
        emb = emb.transpose(-2, -1).contiguous() # (num_utt * batch, num_words, embedding_dim)  -> (num_utt * batch, embedding_dim, num_words) 
        
        convoluted = [F.relu(conv(emb)) for conv in self.convs] 
        pooled = [F.max_pool1d(c, c.size(2)).squeeze() for c in convoluted] 
        concated = torch.cat(pooled, 1)
        features = F.relu(self.fc(self.dropout(concated))) # (num_utt * batch, embedding_dim//2) -> (num_utt * batch, output_size)
        features = features.view(num_utt, batch, -1) # (num_utt * batch, output_size) -> (num_utt, batch, output_size)
        mask = umask.unsqueeze(-1).type(FloatTensor) # (batch, num_utt) -> (batch, num_utt, 1)
        mask = mask.transpose(0, 1) # (batch, num_utt, 1) -> (num_utt, batch, 1)
        mask = mask.repeat(1, 1, self.feature_dim) #  (num_utt, batch, 1) -> (num_utt, batch, output_size)
        features = (features * mask) # (num_utt, batch, output_size) -> (num_utt, batch, output_size)

        return features


class Trans(nn.Module):
    def __init__(self,D_m, n_classes=2, attention=False):
    # def __init__(self, D_m,D_e, D_h, output_dim=7, freeze):
        super(Trans, self).__init__()
        self.dropout   = nn.Dropout(.5)
        encoder_layer = nn.TransformerEncoderLayer(d_model=D_m, nhead=7)
        self.transformer= nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        self.linear = nn.Linear(2534, 2534)
        self.smax_fc = nn.Linear(2534, 1)
        
    def forward(self, U):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions= self.transformer(U)
        #print(emotions.shape)
        
        hidden = F.relu(self.linear(emotions))
        
        hidden = self.dropout(emotions)
        #print(hidden.shape)
        log_prob_hu = self.smax_fc(hidden)
        return log_prob_hu
class Transformer(nn.Module):
    def __init__(self,D_m, D_e, D_h, n_classes=2, attention=False):
    # def __init__(self, D_m,D_e, D_h, output_dim=7, freeze):
        super(Transformer, self).__init__()
        self.dropout   = nn.Dropout(.5)
        encoder_layer = nn.TransformerEncoderLayer(d_model=D_m, nhead=4)
        self.transformer= nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.attention = attention
        
        if self.attention:
            self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        
        self.linear = nn.Linear(2524, 2524)
        self.smax_fc = nn.Linear(2524, 7)
        self.smax_fc1 = nn.Linear(2524, 3)
        
    def forward(self, U, umask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions= self.transformer(U)
        #print(emotions.shape)
        alpha, alpha_f, alpha_b = [], [], []
        
        if self.attention:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            #print(emotions.shape)
            hidden = F.relu(self.linear(emotions))
        
        hidden = self.dropout(emotions)
        #print(hidden.shape)
        log_prob_emo = self.smax_fc(hidden)
        log_prob_senti = self.smax_fc1(hidden)
        return log_prob_senti,log_prob_emo, alpha, alpha_f, alpha_b,hidden


class RMS_Fourier(nn.Module):

    def __fftNd(input, signal_ndim=1, normalized=False, onesided=True, is_rfft=False, is_inverse=False):
    
    # Collect arguments in dictionary for final call
    args = {'input':input, 'signal_ndim':signal_ndim, 'normalized':normalized}
    # Pointer to function to use
    fft_func = torch.fft
    if is_inverse:
        fft_func = torch.ifft

    # If less or equal to 3 dimensions, use pytorch implementation
    if signal_ndim<=3:
        return fft_func(**args)

    # Assign names to dimensions for easier permuting
    dimension_names = ['batch','chan']
    dimension_names.extend([letters[i] for i in range(input.ndim-2)])
    dimension_names[-1] = 'complex'
    input = input.refine_names(*dimension_names)
    
    original_size = input.shape
    dims_ids = [n for n in range(len(original_size))]

    # Set signal dimention to 1, as nD fourier is performed by n sucesive 1D ffts
    out_result = input
    args['signal_ndim'] = 1
    last_dim = 1

    # Iterate dimensions
    for nDim in range(2,len(original_size)-last_dim):
        curr_char = dimension_names[nDim]
        # 1D fft of every dimension indivisually, so atach every other into batch dimension
        new_size = [(dimension_names[i]) for i in range(2,len(original_size)) if dimension_names[i]!=curr_char and dimension_names[i]!='complex'] 
        new_size = ['batch'] + new_size + ['chan',curr_char,'complex']

        # Permute such that all dimensions are stacked to the batch dimension, except the nDim
        middle_result = out_result.align_to(*new_size)

        # Compute view shape to run fft 1D on
        middle_size = list(middle_result.shape)
        batch_size = [middle_result.shape[i] for i in range(middle_result.ndim-2)]
        batch_size = np.prod(batch_size)
        view_size = [batch_size, original_size[1],original_size[nDim], 2]
        # And reshape
        middle_result = middle_result.contiguous().rename(None).view(view_size)

        # Update arguments for fft
        args['input'] = middle_result

        # Check if it is irfft for last dimension
        if is_inverse and is_rfft and nDim == len(original_size)-last_dim-1:
            fft_func = torch.irfft
            # Remove complex dimension
            middle_size = middle_size[:-1]
            new_size = new_size[:-1]
            args['onesided'] = onesided
            if onesided:
                middle_size[-1] += middle_size[-1]//2
        if is_inverse == False and is_rfft and nDim == len(original_size)-last_dim-1:
            fft_func = torch.irfft
            # Remove complex dimension
            middle_size = middle_size[:-1]
            new_size = new_size[:-1]
            args['onesided'] = False

        # Run fft_func
        middle_result = fft_func(**args)

        # Get back to original shape
        out_result = middle_result.view(middle_size).refine_names(*new_size)
        out_result = out_result.align_as(input)
    # Remove dimention names and return result
    return out_result.rename(None)



	### Specific functions, visible to users
	def fftNd(input, signal_ndim=1, normalized=False):
    	return __fftNd(input, signal_ndim=signal_ndim, normalized=normalized)

	def ifftNd(input, signal_ndim=1, normalized=False, signal_sizes=()):
    	return __fftNd(input, signal_ndim=signal_ndim, normalized=normalized, is_inverse=True)

	def rfftNd(input, signal_ndim=1, normalized=False, onesided=True):
    # Simulate rfft with ffts
    	dims = input.ndim * [1]
    	dims.extend([2])
    	input = input.unsqueeze(input.ndim).repeat(dims)
    	input[...,1] = 0
    	result = __fftNd(input, signal_ndim=signal_ndim, normalized=normalized)
    	if onesided:
        	result = result[...,:result.shape[-2]//2+1,:]
    	return result

	def irfftNd(input, signal_ndim=1, normalized=False, onesided=True, signal_sizes=()):
    	result = __fftNd(input, signal_ndim=signal_ndim, normalized=normalized, onesided=onesided, is_rfft=True, is_inverse=True)
    	return result[...,0]
    
    
class E2ELSTMModel(nn.Module):

    def __init__(self, D_e, D_h,
                 vocab_size, embedding_dim=300, 
                 cnn_output_size=100, cnn_filters=50, cnn_kernel_sizes=(3,4,5), cnn_dropout=0.5,
                 n_classes=7, dropout=0.5, attention=False):
        
        super(E2ELSTMModel, self).__init__()

        self.cnn_feat_extractor = CNNFeatureExtractor(vocab_size, embedding_dim, cnn_output_size, cnn_filters, cnn_kernel_sizes, cnn_dropout)
        
        self.dropout   = nn.Dropout(dropout)
        self.attention = attention
        self.lstm = nn.LSTM(input_size=cnn_output_size, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        
        if self.attention:
            self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        
        self.linear = nn.Linear(2*D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)
    
    def init_pretrained_embeddings(self, pretrained_word_vectors):
        self.cnn_feat_extractor.init_pretrained_embeddings_from_numpy(pretrained_word_vectors)

    def forward(self, input_seq, qmask, umask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        U = self.cnn_feat_extractor(input_seq, umask)
        
        emotions, hidden = self.lstm(U)
        alpha, alpha_f, alpha_b = [], [], []
        
        if self.attention:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
            
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            #print("i am helloooooooooooooo",att_emotions)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
        
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        return log_prob, alpha, alpha_f, alpha_b
              



