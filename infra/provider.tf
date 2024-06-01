terraform{
  required_version = "~>1.8.0"
  required_providers {
    aws={
      source="hashicorp/aws"
      version = "~>5.0"
    }
  }
  backend "s3"{
    bucket = "mlops-bucket-cdc"
    key = "tfstate/terraform.tfstate"
    dynamodb_table = "mlopslock"
    region = "eu-west-1"
  }
}
provider "aws" {
  region = "eu-west-1"
}