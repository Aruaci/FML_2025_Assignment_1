from partitioners import partition_dataset, partition_dataset_by_gender, partition_dataset_by_age, partition_dataset_by_education
from torch.utils.data import DataLoader

def load_federated_datasets(x_train, x_test, y_train, y_test, num_clients, batch_size):
    train_partitions = partition_dataset(x_train, y_train, num_clients)
    test_partitions = partition_dataset(x_test, y_test, num_clients)
    
    federated_trainloaders = []
    federated_testloaders = []

    for train_partition, test_partition in zip(train_partitions, test_partitions):
        trainloader = DataLoader(train_partition, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(test_partition, batch_size=batch_size, shuffle=False)
        federated_trainloaders.append(trainloader)
        federated_testloaders.append(testloader)

    return federated_trainloaders, federated_testloaders

def load_federated_datasets_by_gender(
    x_train, x_test, y_train, y_test, train_df, test_df, num_clients, batch_size, gender_column
):
    train_partitions = partition_dataset_by_gender(
        x_train, y_train, gender_column, train_df, num_clients
    )
    test_partitions = partition_dataset_by_gender(
        x_test, y_test, gender_column, test_df, num_clients
    )
    
    federated_trainloaders = [
        DataLoader(partition, batch_size=batch_size, shuffle=True) for partition in train_partitions
    ]
    federated_testloaders = [
        DataLoader(partition, batch_size=batch_size, shuffle=False) for partition in test_partitions
    ]

    return federated_trainloaders, federated_testloaders

def load_federated_datasets_by_age(
    x_train, x_test, y_train, y_test, train_df, test_df, num_clients, batch_size, age_column, scaled_min, scaled_max
):
    train_partitions = partition_dataset_by_age(
        x_train, y_train, train_df, age_column, scaled_min, scaled_max, num_clients
    )
    test_partitions = partition_dataset_by_age(
        x_test, y_test, test_df, age_column, scaled_min, scaled_max, num_clients
    )

    federated_trainloaders = [
        DataLoader(partition, batch_size=batch_size, shuffle=True) for partition in train_partitions
    ]
    federated_testloaders = [
        DataLoader(partition, batch_size=batch_size, shuffle=False) for partition in test_partitions
    ]

    return federated_trainloaders, federated_testloaders

def load_federated_datasets_by_education(
    x_train, x_test, y_train, y_test, train_df, test_df, num_clients, batch_size
):
    train_partitions = partition_dataset_by_education(
        x_train, y_train, train_df, num_clients
    )

    test_partitions = partition_dataset_by_education(
        x_test, y_test, test_df, num_clients
    )

    federated_trainloaders = [
        DataLoader(partition, batch_size=batch_size, shuffle=True) for partition in train_partitions
    ]
    federated_testloaders = [
        DataLoader(partition, batch_size=batch_size, shuffle=False) for partition in test_partitions
    ]

    return federated_trainloaders, federated_testloaders
