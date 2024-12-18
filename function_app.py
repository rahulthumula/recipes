import os
import json
import logging
import azure.functions as func
import azure.durable_functions as df
from azure.storage.blob.aio import BlobServiceClient
from azure.cosmos.aio import CosmosClient
from azure.servicebus.aio import ServiceBusClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.search.documents import SearchClient
from openai import OpenAI
from typing import List, Dict, Any
import mimetypes
import tempfile
from datetime import datetime
from azure.servicebus import ServiceBusMessage

from shared.recipe_parser import process_recipe_folder, RecipeCostCalculator
from shared.retail_price_scraper import MultiRetailerPriceScraper


# Initialize function app
myapp = df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Initialize blob client
blob_service_client = BlobServiceClient.from_connection_string(os.environ["AzureWebJobsStorage"])

async def ensure_container_exists():
    """Ensure the recipes container exists."""
    try:
        container_client = blob_service_client.get_container_client("recipes")
        # Check if container exists
        try:
            await container_client.get_container_properties()
        except Exception:
            # Container doesn't exist, create it
            await container_client.create_container()
        return container_client
    except Exception as e:
        raise

def is_valid_file_type(filename: str) -> bool:
    """Check if file type is allowed."""
    allowed_extensions = {'.pdf', '.xlsx', '.xls', '.csv', '.jpg', '.jpeg', '.png'}
    return os.path.splitext(filename)[1].lower() in allowed_extensions

@myapp.route(route="process-recipes/{index_name}", methods=["POST"])
@myapp.durable_client_input(client_name="client")
async def http_trigger(req: func.HttpRequest, client):
    """HTTP trigger for recipe processing."""
    try:
        # Get and validate index name
        index_name = req.route_params.get('index_name')
        if not index_name:
            return func.HttpResponse(
                json.dumps({"error": "Index name is required"}),
                mimetype="application/json",
                status_code=400
            )

        # Ensure container exists
        await ensure_container_exists()

        # Handle file uploads
        blob_references = []
        max_file_size = 50 * 1024 * 1024  # 50MB limit
        
        for file_name in req.files:
            file = req.files[file_name]
            
            # Validate file type
            if not is_valid_file_type(file.filename):
                return func.HttpResponse(
                    json.dumps({"error": f"Invalid file type: {file.filename}"}),
                    mimetype="application/json",
                    status_code=400
                )

            # Validate file size
            file_content = file.read()
            if len(file_content) > max_file_size:
                return func.HttpResponse(
                    json.dumps({"error": f"File too large: {file.filename}"}),
                    mimetype="application/json",
                    status_code=400
                )
            file.seek(0)  # Reset file pointer

            # Upload to blob storage
            blob_name = f"{index_name}/{file.filename}"
            blob_client = blob_service_client.get_blob_client(
                container="recipes",
                blob=blob_name
            )
            await blob_client.upload_blob(file.stream, overwrite=True)
            blob_references.append({"blob_name": blob_name})

        if not blob_references:
            return func.HttpResponse(
                json.dumps({"error": "No files uploaded"}),
                mimetype="application/json",
                status_code=400
            )

        # Start orchestration
        instance_id = await client.start_new(
            "recipe_orchestrator",
            None,
            {"index_name": index_name, "blobs": blob_references}
        )
        
        return client.create_check_status_response(req, instance_id)

    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": "Internal server error", "details": str(e)}),
            mimetype="application/json",
            status_code=500
        )

@myapp.orchestration_trigger(context_name="context")
def recipe_orchestrator(context):
    """Orchestrator function for recipe processing."""
    try:
        input_data = context.get_input()
        if isinstance(input_data, str):
            input_data = json.loads(input_data)
            
        index_name = input_data.get("index_name")
        blobs = input_data.get("blobs", [])

        if not blobs or not index_name:
            return {
                "status": "failed",
                "message": "Invalid input data",
                "recipe_count": 0
            }

        # Process files in parallel
        tasks = []
        for blob in blobs:
            task = context.call_activity("process_file_activity", {
                "blob": blob,
                "index_name": index_name
            })
            tasks.append(task)

        # Wait for all processing to complete
        results = yield context.task_all(tasks)

        # Validate results
        valid_results = [r for result in results for r in result if r and isinstance(r, dict)]
        
        if not valid_results:
            return {
                "status": "completed",
                "message": "No valid recipes found",
                "recipe_count": 0
            }

        # Store results
        storedata = {
            "index_name": index_name,
            "recipes": valid_results
        }
        
        yield context.call_activity("store_recipes_activity", storedata)

        return {
            "status": "completed",
            "message": f"Processed {len(valid_results)} recipes",
            "recipe_count": len(valid_results)
        }

    except Exception as e:
        return {"status": "failed", "error": str(e)}

@myapp.activity_trigger(input_name="taskinput")
async def process_file_activity(taskinput):
    """Activity function for processing individual files."""
    temp_dir = None
    try:
        blob_info = taskinput.get("blob")
        index_name = taskinput.get("index_name")

        if not blob_info or not index_name:
            return []

        # Initialize clients
        search_client = SearchClient(
            endpoint=os.environ["AZURE_AISEARCH_ENDPOINT"],
            index_name=index_name,
            credential=AzureKeyCredential(os.environ["AZURE_AISEARCH_APIKEY"])
        )
        
        form_client = DocumentAnalysisClient(
            endpoint=os.environ["AZURE_FORM_RECOGNIZER_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["AZURE_FORM_RECOGNIZER_KEY"])
        )
        
        openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        
        blob_client = blob_service_client.get_blob_client(
            container="recipes",
            blob=blob_info["blob_name"]
        )

        # Create temporary directory without async context manager
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, os.path.basename(blob_info["blob_name"]))
        
        # Download file
        download_stream = await blob_client.download_blob()
        with open(temp_path, "wb") as temp_file:
            async for chunk in download_stream.chunks():
                temp_file.write(chunk)

        # Initialize calculator and process recipes
        calculator = RecipeCostCalculator(search_client, openai_client)
        recipes = await process_recipe_folder(
            temp_dir,
            calculator,
            form_client,
            openai_client
        )

        if not recipes or not isinstance(recipes, list):
            return []

        return recipes

    except Exception as e:
        return []
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)

@myapp.activity_trigger(input_name="storeinput")
async def store_recipes_activity(storeinput: Dict[str, Any]) -> Dict:
    """Activity function for storing processed recipes."""
    try:
        recipes = storeinput.get("recipes", [])
        index_name = storeinput.get("index_name")

        if not index_name:
            raise ValueError("Index name is required")

        if not recipes:
            return {
                "status": "completed",
                "message": "No recipes to store",
                "stored_count": 0
            }

        # Store in Cosmos DB
        async with CosmosClient(
            url=os.environ["COSMOS_ENDPOINT"],
            credential=os.environ["COSMOS_KEY"]
        ) as cosmos_client:
            database = cosmos_client.get_database_client("InvoicesDB")
            container = database.get_container_client("Recipes")

            # Store each recipe
            for recipe in recipes:
                recipe["id"] = f"{index_name}_{recipe['recipe_name']}"
                recipe["index_name"] = index_name
                await container.upsert_item(recipe)

            # Send notification to Service Bus
            async with ServiceBusClient.from_connection_string(
                os.environ["ServiceBusConnection"]
            ) as servicebus_client:
                sender = servicebus_client.get_queue_sender("recipe-updates")
                
                message = {
                    "index_name": index_name,
                    "recipe_count": len(recipes),
                    "status": "completed"
                }
                
                await sender.send_messages([
                    ServiceBusMessage(json.dumps(message))
                ])

            return {
                "status": "completed",
                "message": f"Successfully stored {len(recipes)} recipes",
                "stored_count": len(recipes)
            }

    except Exception as e:
        raise