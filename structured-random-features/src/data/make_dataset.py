# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import src.data.preprocessing as preprocess

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    logger.info('Processing sensilla data:')
    preprocess.extract_sensilla_STA(input_filepath, output_filepath)

    logger.info('Processing Marius V1 data w/ whitenoise:')
    preprocess.extract_V1_rf_whitenoise(input_filepath, output_filepath)

    logger.info('Processing Marius V1 data w/ natural images:')
    preprocess.extract_V1_rf_natural_images(input_filepath, output_filepath)

    logger.info('Processing Marius V1 data w/ DHT:')
    preprocess.extract_V1_rf_DHT(input_filepath, output_filepath)

    logger.info('Processing Ringach V1 data:')
    preprocess.extract_V1_rf_Ringach(input_filepath, output_filepath)

    ## Download KMNIST dataset
    logger.info('Downloading KMNIST data:')
    preprocess.download_KMNIST(output_filepath)

    ## Download MNIST dataset
    logger.info('Downloading MNIST data:')
    preprocess.download_MNIST(output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
