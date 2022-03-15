"""Example workflow pipeline script for abalone pipeline.

                                               . -RegisterModel
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""

import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.pytorch import (PyTorch, PyTorchModel)
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.tuner import HyperparameterTuner
from sagemaker.parameter import (
    ContinuousParameter,
    CategoricalParameter
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    TuningStep,
    CacheConfig
)
from sagemaker.workflow.step_collections import RegisterModel


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="MNISTPackageGroup",
    pipeline_name="MNISTPipeline",
    base_job_prefix="MNIST",
    use_hpo=False
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
              
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
    
    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.xlarge")
    training_instance_count = ParameterInteger(name="TrainingInstanceCount", default_value=1)
    training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.g4dn.xlarge")
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
    
    # Cache Config
    # https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-caching.html
    cache_config = CacheConfig(enable_caching=True, expire_after="P7D")
    
    # processing step for feature engineering
    # Docker Registry https://docs.aws.amazon.com/sagemaker/latest/dg/ecr-ap-northeast-2.html
    # 토치비젼으로 전처리 하는 케이스에 사용하기 위해 토치용 이미지를 불러왔다.
    image_uri_processing = sagemaker.image_uris.retrieve(
        framework="pytorch",
        region=region,
        version="1.9.1",
        py_version="py38",
        instance_type=processing_instance_type,
        image_scope="inference",
    )
    # 커스텀 스크립트로 전처리를 하는 경우 스크립트 프로세서를 사용
    script_process = ScriptProcessor(
        image_uri=image_uri_processing,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        sagemaker_session=sagemaker_session,
        role=role,
        base_job_name=f"{base_job_prefix}/script-mnist-process",
    )
    # 스텝으로 등록한다
    # 이미 S3에 데이터가 있는 경우 input을 지정해도 됨
    # output은 버켓에 저장 된다. end of job, continuous 옵션이 있음
    step_process = ProcessingStep(
        name="PreprocessMNISTData",
        processor=script_process,
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train/"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test/"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        cache_config=cache_config
    )
    
    
    # training step for generating model artifacts
    hyperparameters = {'epochs':30,'batch-size':256, 'backend': 'gloo'}
    metric_definitions=[{'Name': 'train:error', 'Regex': 'Train Loss: ([0-9\\.]+)'}, {'Name': 'test:error', 'Regex': 'Test set: Average loss: ([0-9\\.]+)'}]
    mnist_train = PyTorch(
        entry_point="train.py",
        source_dir=BASE_DIR,
        role=role,
        py_version='py38',
        framework_version="1.9.1",
        instance_count=training_instance_count,
        instance_type=training_instance_type,
        hyperparameters=hyperparameters,
        metric_definitions=metric_definitions,
        output_path = f"s3://{default_bucket}/{base_job_prefix}",
        base_job_name=f"{base_job_prefix}/pytorch-mnist-training",
    )
    
    mnist_train_input = {
        "train": TrainingInput(s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri),
        "test": TrainingInput(s3_data=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri),
    }
    mnist_train_dummy_input = {
        "train": TrainingInput(s3_data="s3://sagemaker-ap-northeast-2-238312515155/sm-mnist/script-mnist-process-2022-02-09-08-59-26-681/output/train"),
        "test": TrainingInput(s3_data="s3://sagemaker-ap-northeast-2-238312515155/sm-mnist/script-mnist-process-2022-02-09-08-59-26-681/output/test"),
    }
    model_artifact = None
    
    if use_hpo:
        hyperparameter_ranges = {
            "lr": ContinuousParameter(0.001, 0.1),
            "batch-size": CategoricalParameter([32, 64, 128, 256, 512]),
        }
        objective_metric_name = "test:error"
        objective_type = "Minimize"
        metric_definitions=[{'Name': 'train:error', 'Regex': 'Train Loss: ([0-9\\.]+)'}, {'Name': 'test:error', 'Regex': 'Test set: Average loss: ([0-9\\.]+)'}]
        mnist_tuner = HyperparameterTuner(
            mnist_train,
            objective_metric_name,
            hyperparameter_ranges,
            metric_definitions,
            max_jobs=20,
            max_parallel_jobs=5,
            objective_type=objective_type,
        )
        step_tuning = TuningStep(
            name="TuningMNISTModel",
            tuner=mnist_tuner,
            inputs=mnist_train_input,
        )
        model_artifact=step_tuning.get_top_model_s3_uri(top_k=0, s3_bucket=default_bucket, prefix=base_job_prefix)
    else:
        step_train = TrainingStep(
            name="TrainMNISTModel",
            estimator=mnist_train,
            inputs=mnist_train_input,
            cache_config = CacheConfig(enable_caching=True, expire_after="P7D")
        )
        model_artifact=step_train.properties.ModelArtifacts.S3ModelArtifacts
    
    
    # processing step for evaluation
    script_eval = ScriptProcessor(
        image_uri=image_uri_processing,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        sagemaker_session=sagemaker_session,
        role=role,
        base_job_name=f"{base_job_prefix}/script-mnist-eval",
    )
    evaluation_report = PropertyFile(
        name="MNISTEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    mnist_eval_dummy_model = "s3://sagemaker-ap-northeast-2-238312515155/sm-mnist/pipelines-d6n63bndkfj5-TrainMNISTModel-cltXc4joRm/output"
    mnist_eval_dummy_source = "s3://sagemaker-ap-northeast-2-238312515155/sm-mnist/script-mnist-process-2022-02-08-08-29-18-597/output/test"
    step_eval = ProcessingStep(
        name="EvaluateMNISTModel",
        processor=script_eval,
        inputs=[
            ProcessingInput(
               source=f"{BASE_DIR}/code",
                destination="/opt/ml/processing/input/code/code",
            ),
            ProcessingInput(
                source=model_artifact,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
        property_files=[evaluation_report],
        job_arguments=["--test","/opt/ml/processing/test"]
    )
    
    # register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )
    
#     model_artifact = "s3://sagemaker-ap-northeast-2-238312515155/sm-mnist/pipelines-j9uxu7dtp31j-TrainMNISTModel-MUenCFfmMO/output/model.tar.gz"
    mnist_model = PyTorchModel(
        model_data=model_artifact,
        entry_point="inference.py",
        source_dir=BASE_DIR,
        role=role,
        py_version='py38',
        framework_version="1.9.1",
        sagemaker_session=sagemaker_session
        
    )
    step_register = RegisterModel(
        name="RegisterMNISTModel",
        estimator=mnist_train,
        model_data=model_artifact,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )

    # condition step for evaluating model quality and branching execution
    cond_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step=step_eval,
            property_file=evaluation_report,
            json_path="classification_metrics.nll_loss.value"
        ),
        right=1.0,
    )
    
    step_cond = ConditionStep(
        name="CheckEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )
    
    # pipeline instance
    steps=[]
    if use_hpo:
        steps=[step_process, step_tuning, step_eval, step_cond]
    else:
        steps=[step_process, step_train, step_eval, step_cond]
#     steps=[step_train, step_register]
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            training_instance_count,
            model_approval_status
        ],
        steps=steps,
        sagemaker_session=sagemaker_session,
    )
    return pipeline
