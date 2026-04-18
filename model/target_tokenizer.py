from collections import defaultdict


class OldEnglishTokenizer:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq

        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        self.special_tokens = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
        ]

        self.token_to_id = {}
        self.id_to_token = {}
    #=
    def tokenize(self, text):
        if not text:
            return []
        text = str(text).strip().lower()#get rid of leading and trailing whitespace and convert to lowercase
        return text.split() #split the text into tokens based on whitespace , put it like [] format 

    def fit(self, sentences):
        count = defaultdict(int) #{}
        temp = self.special_tokens.copy()#['<pad>', '<bos>', '<eos>', '<unk>']
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            for token in tokens:
                count[token] += 1 #count the frequency of each token in the sentences
        for token, freq in count.items(): #iterate through the token and its frequency in the count dictionary
            if freq >= self.min_freq:
                temp.append(token)
        for idx, token in enumerate(temp):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        
        
        
        
    def encode(self, text, add_special_tokens=True):#encode the text into a list of token ids
        ans = []
        if add_special_tokens:
            ans.append(self.bos_id) #add the bos token id at the beginning of the list
        tokens = self.tokenize(text)
        #add the token id to the ans list if the token is in the token_to_id dictionary, otherwise add the unk token id
        for token in tokens:
            if token in self.token_to_id:
                ans.append(self.token_to_id[token])
            else:
                ans.append(self.unk_id)
        if add_special_tokens:
            ans.append(self.eos_id) #add the eos token id at the end of the list
        
        return ans
            

    def decode(self, token_ids, skip_special_tokens=True):
        ans = []
        for i in range(len(token_ids)):
            token_id = int(token_ids[i])
            if skip_special_tokens and token_id in [self.pad_id, self.bos_id, self.eos_id]:
                continue
            token = self.id_to_token.get(token_id, self.unk_token)
            ans.append(token)
        return " ".join(ans)
    @property
    def pad_id(self):
        return self.token_to_id[self.pad_token]

    @property
    def bos_id(self):
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self):
        return self.token_to_id[self.eos_token]

    @property
    def unk_id(self):
        return self.token_to_id[self.unk_token]
