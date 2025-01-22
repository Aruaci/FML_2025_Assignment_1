from utils import CustomDataset
from torch.utils.data import random_split

def partition_dataset(features, labels, num_clients):
    dataset = CustomDataset(features, labels)
    dataset_size = len(dataset)
    partition_size = dataset_size // num_clients
    
    lengths = [partition_size] * num_clients
    lengths[-1] += dataset_size % num_clients
    partitions = random_split(dataset, lengths)
    return partitions

def partition_dataset_by_gender(features, labels, gender_column, source_df, num_clients):
    male_indices = source_df[gender_column] == 1.0
    female_indices = source_df[gender_column] == 0.0
    
    male_features = features[male_indices]
    male_labels = labels[male_indices]
    
    female_features = features[female_indices]
    female_labels = labels[female_indices]

    clients_per_group = num_clients // 2

    male_partitions = partition_dataset(male_features, male_labels, clients_per_group)
    female_partitions = partition_dataset(female_features, female_labels, clients_per_group)
    
    partitions = male_partitions + female_partitions

    remaining_clients = num_clients % 2

    if remaining_clients > 0:
        remaining_indices = source_df.sample(n=remaining_clients, random_state=42).index
        remaining_features = features[remaining_indices]
        remaining_labels = labels[remaining_indices]
        remaining_partitions = partition_dataset(remaining_features, remaining_labels, remaining_clients)
        partitions.extend(remaining_partitions)

    return partitions

def partition_dataset_by_age(features, labels, source_df, age_column, scaled_min, scaled_max, num_clients):
    age_filtered_indices = (source_df[age_column] >= scaled_min) & (source_df[age_column] <= scaled_max)
    age_filtered_features = features[age_filtered_indices]
    age_filtered_labels = labels[age_filtered_indices]

    # Partition this data for 3 specific clients
    age_partitions = partition_dataset(age_filtered_features, age_filtered_labels, 3)

    # Filter out the age-specific data for the remaining dataset
    remaining_indices = ~age_filtered_indices
    remaining_features = features[remaining_indices]
    remaining_labels = labels[remaining_indices]

    # Partition the remaining data for the other clients
    remaining_clients = num_clients - 3
    remaining_partitions = partition_dataset(remaining_features, remaining_labels, remaining_clients)

    # Combine the partitions
    partitions = age_partitions + remaining_partitions

    return partitions

def partition_dataset_by_education(features, labels, source_df, num_clients = 10):
    # Define education levels for each group
    hs_or_below_columns = ['education_11th', 'education_12th', 'education_1st-4th', 
                           'education_5th-6th', 'education_7th-8th', 'education_9th', 
                           'education_HS-grad', 'education_Some-college']
    bachelors_or_higher_columns = ['education_Bachelors', 'education_Masters', 
                                   'education_Doctorate', 'education_Prof-school']
    
    # Filter for HS or below (including Some-college)
    hs_or_below_indices = source_df[hs_or_below_columns].sum(axis=1) > 0
    hs_or_below_features = features[hs_or_below_indices]
    hs_or_below_labels = labels[hs_or_below_indices]
    
    # Partition this data for the first 5 clients
    num_clients_hs_or_below = num_clients // 2
    hs_or_below_partitions = partition_dataset(hs_or_below_features, 
                                               hs_or_below_labels, 
                                               num_clients_hs_or_below)
    
    # Filter for Bachelors or higher
    bachelors_or_higher_indices = source_df[bachelors_or_higher_columns].sum(axis=1) > 0
    bachelors_or_higher_features = features[bachelors_or_higher_indices]
    bachelors_or_higher_labels = labels[bachelors_or_higher_indices]
    
    # Partition this data for the last 5 clients
    num_clients_bachelors_or_higher = num_clients // 2
    bachelors_or_higher_partitions = partition_dataset(bachelors_or_higher_features, 
                                                       bachelors_or_higher_labels, 
                                                       num_clients_bachelors_or_higher)
    
    # Combine the partitions
    partitions = hs_or_below_partitions + bachelors_or_higher_partitions

    return partitions
