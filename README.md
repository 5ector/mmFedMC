# mmFedMC

We propose multimodal Federated learning with joint Modality and Client selection (mmFedMC), an mmFL methodology tailored for clients with heterogeneously absent modalities. 

## Overview

An overview of our proposed method and comparison with traditional mmFL are given in Fig. 1. We propose a decision-level fusion approach, where predictions from global modality models are used as input to the individual local ensemble model in each client. This allows for the independent deployment of the modality models in various application scenarios, accommodating situations where a client may possess a different set of modalities.

## Key Features

- **Decision-level Fusion**: Utilizes predictions from global modality models as input to local ensemble models.
- **Classical ML Models**: Employs models like Random Forest for the ensemble, reducing computational overhead and improving interpretability.
- **Selective Modality Communication**: Takes into account factors such as modality performance, communication overhead, and recency.
- **Selective Client Uploading**: Optimizes efficiency and effectiveness by ranking clients based on local loss of modality models.

## How to Use

1. **Clone the repository**:
    ```bash
    git clone https://github.com/5ector/mmFedMC.git
    ```
2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the project**:
    ```bash
    python main.py
    ```

## Contribution

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project does not currently have a license.

## Contact

For more information, please visit the [repository](https://github.com/5ector/mmFedMC) or contact the repository owner [5ector](https://github.com/5ector).