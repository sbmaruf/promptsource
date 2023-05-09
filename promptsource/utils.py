# coding=utf-8
import os

import datasets
import requests

from promptsource import DEFAULT_PROMPTSOURCE_CACHE_HOME
from promptsource.templates import INCLUDED_USERS


def removeHyphen(example):
    example_clean = {}
    for key in example.keys():
        if "-" in key:
            new_key = key.replace("-", "_")
            example_clean[new_key] = example[key]
        else:
            example_clean[key] = example[key]
    example = example_clean
    return example


def renameDatasetColumn(dataset):
    col_names = dataset.column_names
    for cols in col_names:
        if "-" in cols:
            dataset = dataset.rename_column(cols, cols.replace("-", "_"))
    return dataset


#
# Helper functions for datasets library
#


def get_dataset_builder(path, conf=None):
    "Get a dataset builder from name and conf."
    module_path = datasets.load.dataset_module_factory(path)
    builder_cls = datasets.load.import_main_class(module_path.module_path, dataset=True)
    if conf:
        builder_instance = builder_cls(name=conf, cache_dir=None, hash=module_path.hash)
    else:
        builder_instance = builder_cls(cache_dir=None, hash=module_path.hash)
    return builder_instance


def get_dataset(path, conf=None):
    "Get a dataset from name and conf."
    try:
        return datasets.load_dataset(path, conf, ignore_verifications=True)
    except datasets.builder.ManualDownloadError:
        cache_root_dir = (
            os.environ["PROMPTSOURCE_MANUAL_DATASET_DIR"]
            if "PROMPTSOURCE_MANUAL_DATASET_DIR" in os.environ
            else DEFAULT_PROMPTSOURCE_CACHE_HOME
        )
        data_dir = f"{cache_root_dir}/{path}" if conf is None else f"{cache_root_dir}/{path}/{conf}"
        try:
            return datasets.load_dataset(
                path,
                conf,
                data_dir=data_dir,
                ignore_verifications=True
            )
        except Exception as err:
            raise err
    except Exception as err:
        raise err


def get_dataset_confs(path):
    "Get the list of confs for a dataset."
    module_path = datasets.load.dataset_module_factory(path).module_path
    # Get dataset builder class from the processing script
    builder_cls = datasets.load.import_main_class(module_path, dataset=True)
    # Instantiate the dataset builder
    confs = builder_cls.BUILDER_CONFIGS
    if confs and len(confs) > 1:
        return confs
    return []


def render_features(features):
    """Recursively render the dataset schema (i.e. the fields)."""
    if isinstance(features, dict):
        return {k: render_features(v) for k, v in features.items()}
    if isinstance(features, datasets.features.ClassLabel):
        return features.names

    if isinstance(features, datasets.features.Value):
        return features.dtype

    if isinstance(features, datasets.features.Sequence):
        return {"[]": render_features(features.feature)}
    return features


#
# Loads dataset information
#


def filter_datasets():
    """
    Filter datasets from HuggingFace API.
    Also includes the datasets of any users listed in INCLUDED_USERS
    """
    filtered_datasets = []

    response = requests.get("https://huggingface.co/api/datasets?full=true")
    tags = response.json()
    while "next" in response.links:
        # Handle pagination of `/api/datasets` endpoint
        response = requests.get(response.links["next"]["url"])
        tags += response.json()

    for dataset in tags:
        dataset_name = dataset["id"]

        is_community_dataset = "/" in dataset_name
        if is_community_dataset:
            user = dataset_name.split("/")[0]
            if user in INCLUDED_USERS:
                filtered_datasets.append(dataset_name)
            continue

        filtered_datasets.append(dataset_name)

    return sorted(filtered_datasets)

SERIES_A_DATASET_NAME_DICT = {
	"udhr": None,
	"AmazonScience/mintaka": None,
	"xcsr": [
		'X-CSQA-en', 
		'X-CSQA-zh', 
		'X-CSQA-de', 
		'X-CSQA-es', 
		'X-CSQA-fr', 
		'X-CSQA-it', 
		'X-CSQA-jap', 
		'X-CSQA-nl', 
		'X-CSQA-pl', 
		'X-CSQA-pt', 
		'X-CSQA-ru', 
		'X-CSQA-ar', 
		'X-CSQA-vi', 
		'X-CSQA-hi', 
		'X-CSQA-sw', 
		'X-CSQA-ur', 
		'X-CODAH-en', 
		'X-CODAH-zh', 
		'X-CODAH-de', 
		'X-CODAH-es', 
		'X-CODAH-fr', 
		'X-CODAH-it', 
		'X-CODAH-jap', 
		'X-CODAH-nl', 
		'X-CODAH-pl', 
		'X-CODAH-pt', 
		'X-CODAH-ru', 
		'X-CODAH-ar', 
		'X-CODAH-vi', 
		'X-CODAH-hi', 
		'X-CODAH-sw', 
		'X-CODAH-ur'
	],
	"shmuhammad/AfriSenti-twitter-sentiment": [
		'amh', 
		'hau', 
		'ibo',
		'arq', 
		'ary', 
		'yor', 
		'por', 
		'twi', 
		'tso', 
		'tir', 
		'pcm', 
		'kin', 
		'swa'
	], # orm is not workin
	"indonlp/NusaX-senti": [
		'ace', 
		'ban', 
		'bjn', 
		'bug',
		'eng',
		'ind',
		'jav', 
		'mad', 
		'min', 
		'nij', 
		'sun', 
		'bbc'
	],
	"masakhane/masakhanews": [
		'amh', 
		'eng', 
		'fra', 
		'hau', 
		'ibo', 
		'lin', 
		'lug', 
		'orm', 
		'pcm', 
		'run', 
		'sna', 
		'som', 
		'swa', 
		'tir', 
		'xho', 
		'yor'
	],
	"papluca/language-identification": [
		'wikipedia-zero-shot', 
		'wikipedia-zero-shot.af', 
		'wikipedia-zero-shot.ar', 
		'wikipedia-zero-shot.be', 
		'wikipedia-zero-shot.bg', 
		'wikipedia-zero-shot.bn',
		'wikipedia-zero-shot.ca',
		'wikipedia-zero-shot.cs', 
		'wikipedia-zero-shot.da', 
		'wikipedia-zero-shot.de', 
		'wikipedia-zero-shot.el', 
		'wikipedia-zero-shot.en',
		'wikipedia-zero-shot.es',
		'wikipedia-zero-shot.fa', 
		'wikipedia-zero-shot.fi',
		'wikipedia-zero-shot.fr',
		'wikipedia-zero-shot.he',
		'wikipedia-zero-shot.hi',
		'wikipedia-zero-shot.hu',
		'wikipedia-zero-shot.id',
		'wikipedia-zero-shot.it',
		'wikipedia-zero-shot.ja',
		'wikipedia-zero-shot.ko',
		'wikipedia-zero-shot.ml',
		'wikipedia-zero-shot.mr',
		'wikipedia-zero-shot.ms',
		'wikipedia-zero-shot.nl',
		'wikipedia-zero-shot.no',
		'wikipedia-zero-shot.pl',
		'wikipedia-zero-shot.pt',
		'wikipedia-zero-shot.ro',
		'wikipedia-zero-shot.ru',
		'wikipedia-zero-shot.si',
		'wikipedia-zero-shot.sk',
  		'wikipedia-zero-shot.sl',
		'wikipedia-zero-shot.sr', 
		'wikipedia-zero-shot.sv', 
		'wikipedia-zero-shot.sw',
		'wikipedia-zero-shot.ta', 
		'wikipedia-zero-shot.te',
		'wikipedia-zero-shot.th',
		'wikipedia-zero-shot.tr',
		'wikipedia-zero-shot.uk',
		'wikipedia-zero-shot.vi',
		'wikipedia-zero-shot.zh',
		'wikinews-zero-shot',
		'wikinews-zero-shot.ar',
		'wikinews-zero-shot.cs',
		'wikinews-zero-shot.de',
		'wikinews-zero-shot.en',
		'wikinews-zero-shot.es',
		'wikinews-zero-shot.fi', 
		'wikinews-zero-shot.fr',
		'wikinews-zero-shot.it',
		'wikinews-zero-shot.ja',
		'wikinews-zero-shot.ko',
		'wikinews-zero-shot.nl',
		'wikinews-zero-shot.no',
		'wikinews-zero-shot.pl',
		'wikinews-zero-shot.pt',
		'wikinews-zero-shot.ru',
		'wikinews-zero-shot.sr',
		'wikinews-zero-shot.sv',
		'wikinews-zero-shot.ta',
		'wikinews-zero-shot.tr',
		'wikinews-zero-shot.uk',
		'wikinews-zero-shot.zh',
		'wikinews-cross-domain', 
		'wikinews-cross-domain.ar',
		'wikinews-cross-domain.bg',
		'wikinews-cross-domain.ca',
		'wikinews-cross-domain.cs',
		'wikinews-cross-domain.de',
		'wikinews-cross-domain.el',
		'wikinews-cross-domain.en',
		'wikinews-cross-domain.es',
		'wikinews-cross-domain.fi',
		'wikinews-cross-domain.fr',
		'wikinews-cross-domain.he', 
		'wikinews-cross-domain.hu', 
		'wikinews-cross-domain.it', 
		'wikinews-cross-domain.ja',
  		'wikinews-cross-domain.ko',
		'wikinews-cross-domain.nl',
		'wikinews-cross-domain.no',
  		'wikinews-cross-domain.pl',
		'wikinews-cross-domain.pt',
		'wikinews-cross-domain.ro',
  		'wikinews-cross-domain.ru',
		'wikinews-cross-domain.sr',
		'wikinews-cross-domain.sv',
		'wikinews-cross-domain.ta',
	 	'wikinews-cross-domain.tr',
	  	'wikinews-cross-domain.uk', 
		'wikinews-cross-domain.zh'
	],
	"adithya7/xlel_wd": [
		'wikipedia-zero-shot', 
		'wikipedia-zero-shot.af', 
		'wikipedia-zero-shot.ar', 
		'wikipedia-zero-shot.be', 
		'wikipedia-zero-shot.bg',
		'wikipedia-zero-shot.bn',
		'wikipedia-zero-shot.ca',
		'wikipedia-zero-shot.cs',
		'wikipedia-zero-shot.da',
		'wikipedia-zero-shot.de',
		'wikipedia-zero-shot.el',
		'wikipedia-zero-shot.en',
		'wikipedia-zero-shot.es',
		'wikipedia-zero-shot.fa',
		'wikipedia-zero-shot.fi',
		'wikipedia-zero-shot.fr',
		'wikipedia-zero-shot.he',
		'wikipedia-zero-shot.hi',
		'wikipedia-zero-shot.hu', 
		'wikipedia-zero-shot.id',
		'wikipedia-zero-shot.it',
		'wikipedia-zero-shot.ja',
		'wikipedia-zero-shot.ko',
		'wikipedia-zero-shot.ml',
		'wikipedia-zero-shot.mr',
		'wikipedia-zero-shot.ms',
		'wikipedia-zero-shot.nl',
		'wikipedia-zero-shot.no', 
		'wikipedia-zero-shot.pl',
		'wikipedia-zero-shot.pt',
		'wikipedia-zero-shot.ro',
		'wikipedia-zero-shot.ru',
		'wikipedia-zero-shot.si',
		'wikipedia-zero-shot.sk',
		'wikipedia-zero-shot.sl',
		'wikipedia-zero-shot.sr',
		'wikipedia-zero-shot.sv',
		'wikipedia-zero-shot.sw',
		'wikipedia-zero-shot.ta',
		'wikipedia-zero-shot.te',
		'wikipedia-zero-shot.th', 
		'wikipedia-zero-shot.tr',
		'wikipedia-zero-shot.uk',
	 	'wikipedia-zero-shot.vi', 
		'wikipedia-zero-shot.zh',
	 	'wikinews-zero-shot',
	  	'wikinews-zero-shot.ar',
	   	'wikinews-zero-shot.cs',
		'wikinews-zero-shot.de',
		'wikinews-zero-shot.en',
		'wikinews-zero-shot.es',
		'wikinews-zero-shot.fi',
		'wikinews-zero-shot.fr',
		'wikinews-zero-shot.it',
		'wikinews-zero-shot.ja',
		'wikinews-zero-shot.ko',
		'wikinews-zero-shot.nl', 
		'wikinews-zero-shot.no', 
		'wikinews-zero-shot.pl',
	 	'wikinews-zero-shot.pt', 
		'wikinews-zero-shot.ru', 
		'wikinews-zero-shot.sr',
		'wikinews-zero-shot.sv', 
		'wikinews-zero-shot.ta',
	 	'wikinews-zero-shot.tr',
	  	'wikinews-zero-shot.uk',
		'wikinews-zero-shot.zh',
		'wikinews-cross-domain',
		'wikinews-cross-domain.ar',
		'wikinews-cross-domain.bg',
		'wikinews-cross-domain.ca',
		'wikinews-cross-domain.cs',
	 	'wikinews-cross-domain.de',
		'wikinews-cross-domain.el',
	 	'wikinews-cross-domain.en',
	  	'wikinews-cross-domain.es',
	   	'wikinews-cross-domain.fi',
		'wikinews-cross-domain.fr', 
		'wikinews-cross-domain.he', 
		'wikinews-cross-domain.hu', 
		'wikinews-cross-domain.it', 
		'wikinews-cross-domain.ja', 
		'wikinews-cross-domain.ko', 
		'wikinews-cross-domain.nl', 
		'wikinews-cross-domain.no', 
		'wikinews-cross-domain.pl', 
		'wikinews-cross-domain.pt', 
		'wikinews-cross-domain.ro', 
		'wikinews-cross-domain.ru', 
		'wikinews-cross-domain.sr', 
		'wikinews-cross-domain.sv', 
		'wikinews-cross-domain.ta', 
		'wikinews-cross-domain.tr', 
		'wikinews-cross-domain.uk', 
		'wikinews-cross-domain.zh'
	],
	"sbmaruf/forai_ml-ted_talk_iwslt": [
		'eu_ca_2014', 
	 	'eu_ca_2015', 
	  	'eu_ca_2016', 
	   	'nl_en_2014', 
		'nl_en_2015', 
		'nl_en_2016', 
		'nl_hi_2014', 
		'nl_hi_2015', 
		'nl_hi_2016', 
		'de_ja_2014', 
		'de_ja_2015', 
		'de_ja_2016', 
		'fr-ca_hi_2014', 
		'fr-ca_hi_2015', 
		'fr-ca_hi_2016'
	],
	"sbmaruf/forai_ml_masakhane_mafand":[
		'en-amh', 
		'en-hau', 
		'en-ibo', 
		'en-kin', 
		'en-lug', 
		'en-nya', 
		'en-pcm', 
		'en-sna', 
		'en-swa', 
		'en-tsn', 
		'en-twi', 
		'en-xho', 
		'en-yor', 
		'en-zul', 
		'fr-bam', 
		'fr-bbj', 
		'fr-ewe', 
		'fr-fon', 
		'fr-mos', 
		'fr-wol'
	],
	"exams":[
		'alignments', 
		'multilingual', 
		'multilingual_with_para', 
		'crosslingual_test', 
		'crosslingual_with_para_test', 
		'crosslingual_bg', 
		'crosslingual_with_para_bg', 
		'crosslingual_hr', 
		'crosslingual_with_para_hr', 
		'crosslingual_hu', 
		'crosslingual_with_para_hu', 
		'crosslingual_it', 
		'crosslingual_with_para_it', 
		'crosslingual_mk', 
		'crosslingual_with_para_mk', 
		'crosslingual_pl', 
		'crosslingual_with_para_pl', 
		'crosslingual_pt', 
		'crosslingual_with_para_pt', 
		'crosslingual_sq', 
		'crosslingual_with_para_sq', 
		'crosslingual_sr', 
		'crosslingual_with_para_sr', 
		'crosslingual_tr', 
		'crosslingual_with_para_tr', 
		'crosslingual_vi', 
		'crosslingual_with_para_vi'
	],
	"allenai/soda": None, 
	"arabic_billion_words":[
		'Alittihad', 
		'Almasryalyoum', 
		'Almustaqbal', 
		'Alqabas', 
		'Echoroukonline', 
		'Ryiadh', 
		'Sabanews', 
		'SaudiYoum', 
		'Techreen', 
		'Youm7'
	]
}


CUSTOM_DATASET = list(SERIES_A_DATASET_NAME_DICT.keys())

def list_datasets():
    """Get all the datasets to work with."""
    dataset_list = filter_datasets()
    dataset_list += CUSTOM_DATASET
    dataset_list = list(set(dataset_list))
    dataset_list.sort(key=lambda x: x.lower())
    return dataset_list
