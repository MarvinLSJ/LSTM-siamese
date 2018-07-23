from torch.utils.data import Dataset
from collections import Counter


class myDS(Dataset):

    def __init__(self, df, all_sents):
        # Assign vocabularies.
        self.s1 = df['s1'].tolist()
        self.s2 = df['s2'].tolist()
        self.label = df['label'].tolist()
        self.vocab = Vocab(all_sents, sos_token='<sos>', eos_token='<eos>', unk_token='<unk>')

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # Split sentence into words.
        s1_words = self.s1[idx].split()
        s2_words = self.s2[idx].split()

        # Add <SOS> and <EOS> tokens.
        s1_words = [self.vocab.sos_token] + s1_words + [self.vocab.eos_token]
        s2_words = [self.vocab.sos_token] + s2_words + [self.vocab.eos_token]

        # Lookup word ids in vocabularies.
        s1_ids = [self.vocab.word2id(word) for word in s1_words]
        s2_ids = [self.vocab.word2id(word) for word in s2_words]
        label = self.label[idx]

        return s1_ids, s2_ids, label

class mytestDS(Dataset):

    def __init__(self, df, all_sents):
        # Assign vocabularies.
        self.s1 = df['s1'].tolist()
        self.s2 = df['s2'].tolist()
        self.vocab = Vocab(all_sents, sos_token='<sos>', eos_token='<eos>', unk_token='<unk>')

    def __len__(self):
        return len(self.s1)

    def __getitem__(self, idx):
        # Split sentence into words.
        s1_words = self.s1[idx].split()
        s2_words = self.s2[idx].split()

        # Add <SOS> and <EOS> tokens.
        s1_words = [self.vocab.sos_token] + s1_words + [self.vocab.eos_token]
        s2_words = [self.vocab.sos_token] + s2_words + [self.vocab.eos_token]

        # Lookup word ids in vocabularies.
        s1_ids = [self.vocab.word2id(word) for word in s1_words]
        s2_ids = [self.vocab.word2id(word) for word in s2_words]

        return s1_ids, s2_ids

class Vocab(object):
    def __init__(self, all_sents, max_size=None, sos_token=None, eos_token=None, unk_token=None):
        """Initialize the vocabulary.
        Args:
            iter: An iterable which produces sequences of tokens used to update
                the vocabulary.
            max_size: (Optional) Maximum number of tokens in the vocabulary.
            sos_token: (Optional) Token denoting the start of a sequence.
            eos_token: (Optional) Token denoting the end of a sequence.
            unk_token: (Optional) Token denoting an unknown element in a
                sequence.
        """
        self.max_size = max_size
        self.pad_token = '<pad>'
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        # Add special tokens.
        id2word = [self.pad_token]
        if sos_token is not None:
            id2word.append(self.sos_token)
        if eos_token is not None:
            id2word.append(self.eos_token)
        if unk_token is not None:
            id2word.append(self.unk_token)

        # Update counter with token counts.
        counter = Counter()
        for x in all_sents:
            counter.update(x.split())

        # Extract lookup tables.
        if max_size is not None:
            counts = counter.most_common(max_size)
        else:
            counts = counter.items()
            counts = sorted(counts, key=lambda x: x[1], reverse=True)
        words = [x[0] for x in counts]
        id2word.extend(words)
        word2id = {x: i for i, x in enumerate(id2word)}

        self._id2word = id2word
        self._word2id = word2id

    def __len__(self):
        return len(self._id2word)

    def word2id(self, word):
        """Map a word in the vocabulary to its unique integer id.
        Args:
            word: Word to lookup.
        Returns:
            id: The integer id of the word being looked up.
        """
        if word in self._word2id:
            return self._word2id[word]
        elif self.unk_token is not None:
            return self._word2id[self.unk_token]
        else:
            raise KeyError('Word "%s" not in vocabulary.' % word)

    def id2word(self, id):
        """Map an integer id to its corresponding word in the vocabulary.
        Args:
            id: Integer id of the word being looked up.
        Returns:
            word: The corresponding word.
        """
        return self._id2word[id]
