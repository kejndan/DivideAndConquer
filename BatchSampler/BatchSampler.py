import torch
import numpy as np


class BatchSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(self, dataset, batch_size, m=4):
        self.m = m
        self.dataset = dataset
        self.batch_size = batch_size

        self.prod_labels = np.array(dataset.labels)
        self.unique_prod_labels = list(set(self.prod_labels))

        self.prod_label_indices_within_cur_label = self.get_all_prod_labels_indices_with_cur_label()
        self.nb_classes_to_pick = batch_size // m

        self.indices_real_from_class = self.get_all_prod_labels_indices_with_cur_label()
        self.indices_taken_from_class = np.zeros(len(self.unique_prod_labels))

    def get_all_prod_labels_indices_with_cur_label(self):
        out = []
        for c in self.unique_prod_labels:
            prod_label_indices = np.where(self.prod_labels == c)[0]
            np.random.shuffle(prod_label_indices)
            out.append(prod_label_indices)
        return out

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        for _ in range(len(self.dataset) // self.batch_size):
            if len(self.unique_prod_labels) > self.nb_classes_to_pick:
                replace = False
            else:
                replace = True
            chosen_classes = np.random.choice(
                len(self.unique_prod_labels), self.nb_classes_to_pick,
                replace=replace)
            out = []
            for cl in chosen_classes:
                if self.indices_taken_from_class[cl] + self.m \
                        < len(self.indices_real_from_class[cl]):
                    st = int(self.indices_taken_from_class[cl])
                    chosen_indices = self.indices_real_from_class[cl][st:st + self.m]
                else:
                    chosen_indices = np.random.choice(self.indices_real_from_class[cl],
                                                      self.m,
                                                      replace=len(self.indices_real_from_class[cl]) < self.m)
                out.extend(chosen_indices)
                self.indices_taken_from_class[cl] += self.m

                if self.indices_taken_from_class[cl] + self.m \
                        > len(self.indices_real_from_class[cl]):
                    np.random.shuffle(self.indices_real_from_class[cl])
                    self.indices_taken_from_class[cl] = 0
            yield out
