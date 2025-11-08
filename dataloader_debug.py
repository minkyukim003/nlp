from nlp.dataloader_debug import MyDataset

def main():
    print("Checking dataset read in.")
    dataset = MyDataset("./data/test_raw_data.csv")

    print(f"The length of the dataset is: {len(dataset)}.")

    num_samples = 5
    print(f"Number of samples tested is {num_samples}.")
    for i in range(num_samples):
        content, label, seq_len = dataset[i]
        print(f"Sample {i}:")
        print("Text:", content[:100], "...") 
        print("Label:", label.item())
        print("Seq_len:", seq_len.item())
        print("-" * 50)
    return 0

if __name__ == '__main__':
    main()