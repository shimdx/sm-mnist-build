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

from sagemaker.pytorch import PyTorch
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
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
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
    if use_hpo:
        print(f"use_hop True:{use_hpo}")
    else:
        print(f"use_hop False:{use_hpo}")
              
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
    
    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.xlarge")
    training_instance_count = ParameterInteger(name="TrainingInstanceCount", default_value=1)
    training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.g4dn.xlarge")
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
    
    
    # processing step for feature engineering
    # Docker Registry https://docs.aws.amazon.com/sagemaker/latest/dg/ecr-ap-northeast-2.html
    image_uri_processing = sagemaker.image_uris.retrieve(
        framework="pytorch",
        region=region,
        version="1.9.1",
        py_version="py38",
        instance_type=processing_instance_type,
        image_scope="inference",
    )
    script_process = ScriptProcessor(
        image_uri=image_uri_processing,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-mnist-process",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    step_process = ProcessingStep(
        name="PreprocessMNISTData",
        processor=script_process,
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train/"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test/"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py")
    )
    
    
    # training step for generating model artifacts
    hyperparameters = {'epochs':10,'batch-size':256, 'backend': 'gloo'}
    mnist_train = PyTorch(
        entry_point="train.py",
        source_dir=BASE_DIR,
        role=role,
        py_version='py38',
        framework_version="1.9.1",
        instance_count=training_instance_count,
        instance_type=training_instance_type,
        hyperparameters=hyperparameters,
        output_path = f"s3://{default_bucket}/{base_job_prefix}",
        base_job_name=f"{base_job_prefix}/pytorch-mnist-training",
    )

    step_train = TrainingStep(
        name="TrainMNISTModel",
        estimator=mnist_train,
        inputs={
            "train": TrainingInput(
#                 s3_data="s3://sagemaker-ap-northeast-2-238312515155/sm-mnist/script-mnist-process-2022-02-08-08-29-18-597/output/train"
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
            ),
            "test": TrainingInput(
#                 s3_data="s3://sagemaker-ap-northeast-2-238312515155/sm-mnist/script-mnist-process-2022-02-08-08-29-18-597/output/test"
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
            ),
        },
    )
    
    
    # processing step for evaluation
    script_eval = ScriptProcessor(
        image_uri=image_uri_processing,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-mnist-eval",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    evaluation_report = PropertyFile(
        name="MNISTEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="EvaluateMNISTModel",
        processor=script_eval,
        inputs=[
            ProcessingInput(
               source=f"{BASE_DIR}/code",
                destination="/opt/ml/processing/input/code/code",
            ),
            ProcessingInput(
#                source="s3://sagemaker-ap-northeast-2-238312515155/sm-mnist/pipelines-d6n63bndkfj5-TrainMNISTModel-cltXc4joRm/output",
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
#                 source="s3://sagemaker-ap-northeast-2-238312515155/sm-mnist/script-mnist-process-2022-02-08-08-29-18-597/output/test",
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
    step_register = RegisterModel(
        name="RegisterMNISTModel",
        estimator=mnist_train,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/plain"],
        response_types=["text/plain"],
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
            json_path="regression_metrics.nll_loss.value"
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
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            training_instance_count,
            model_approval_status
        ],
        steps=[step_process, step_train, step_eval, step_cond],
#         steps=[step_eval],
        sagemaker_session=sagemaker_session,
    )
    return pipeline
