import os
import json
import logging
import pandas as pd
from decimal import Decimal
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from openai import OpenAI
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
import pint
import asyncio
import time
from .retail_price_scraper import MultiRetailerPriceScraper
import mimetypes

# Initialize the unit registry
ureg = pint.UnitRegistry()

# Load environment variables
load_dotenv()
key = os.getenv("OPENAI_API_KEY")
class IngredientMatcher:
    """Smart ingredient matching with vector search and GPT selection."""
    
    def __init__(self, search_client, openai_client):
        self.search_client = search_client
        self.openai_client = openai_client
        self.price_scraper = MultiRetailerPriceScraper()
        
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.openai_client.embeddings.create(
                    input=text.strip(),
                    model="text-embedding-ada-002"
                )
                return response.data[0].embedding
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)
    
    def check_name_similarity(self, ingredient_name: str, match_name: str) -> float:
        """Check name similarity using GPT."""
        try:
            prompt = f"""Rate the similarity between these two ingredient names based ONLY on their meaning and culinary usage.
Rules:
- Exact matches or plural forms (e.g. 'tomato' vs 'tomatoes') = 100
- Same ingredient different form (e.g. 'garlic' vs 'garlic powder') = 80
- Common substitutes (e.g. 'vegetable oil' vs 'canola oil') = 70
- Different but related items (e.g. 'cream' vs 'milk') = 50
- Different items = 0

Ingredient 1: {ingredient_name}
Ingredient 2: {match_name}

Return ONLY a number between 0-100. No other text."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at comparing ingredient names."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            similarity = float(response.choices[0].message.content.strip())
            return max(0, min(100, similarity)) / 100  # Normalize to 0-1
            
        except Exception as e:
            return 0
    
    def select_best_match_with_gpt(self, ingredient: Dict, matches: List[Dict]) -> Optional[Dict]:
        """Use GPT to select the best matching inventory item."""
        try:
            # First check name similarity for all matches
            scored_matches = []
            for match in matches:
                similarity = self.check_name_similarity(ingredient['item'], match['inventory_item'])
                if similarity >= 0.8:  # Only consider matches with 80% or higher name similarity
                    scored_matches.append({**match, 'similarity': similarity})
            
            if not scored_matches:
                return None
                
            # Format remaining matches for final selection
            matches_text = "\n".join(
                f"{idx+1}. Item: {match['inventory_item']} (Similarity: {match['similarity']*100:.1f}%)\n"
                f"   Unit: {match['measured_in']}\n"
                f"   Supplier: {match.get('supplier_name', 'Unknown')}\n"
                f"   Cost per unit: ${float(match['cost_per_unit']):.2f}"
                for idx, match in enumerate(scored_matches)
            )

            prompt = f"""Select the best matching inventory item for a recipe ingredient.
Consider ONLY:
1. Unit compatibility
2. Cost reasonableness

All these items have passed name similarity check.

Recipe Ingredient: {ingredient['item']}
Amount Needed: {ingredient['amount']} {ingredient['unit']}

Available matches:
{matches_text}

Return ONLY the number (1-{len(scored_matches)}) of the best match.
If none are suitable due to unit or cost, return 0."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at matching recipe ingredients to inventory items."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            try:
                selected_idx = int(response.choices[0].message.content.strip()) - 1
                if selected_idx >= 0 and selected_idx < len(scored_matches):
                    match = scored_matches[selected_idx]
                    return {
                        'inventory_item': match['inventory_item'],
                        'cost_per_unit': Decimal(str(match['cost_per_unit'])),
                        'unit': match['measured_in'],
                        'supplier': match.get('supplier_name', 'Unknown'),
                        'similarity': match['similarity']
                    }
            except (ValueError, IndexError):
                return None

        except Exception as e:
            return None
    
    def vector_search(self, ingredient: Dict) -> Optional[Dict]:
        """Search for ingredient match using vector search and GPT selection."""
        try:
            embedding = self.get_embedding(ingredient['item'])
            
            vector_query = VectorizedQuery(
                vector=embedding,
                k_nearest_neighbors=5,  # Increased to get more potential matches
                fields="inventory_item_vector"
            )
            results = list(self.search_client.search(
                search_text=None,
                vector_queries=[vector_query],
                select=["inventory_item", "cost_per_unit", "measured_in", "supplier_name"],
                top=5
            ))
            if not results:
                return None
            
            return self.select_best_match_with_gpt(ingredient, results)
            
        except Exception as e:
            return None

    async def get_best_match(self, ingredient: Dict) -> Optional[Dict]:
        """Get best match from inventory or retail sources."""
        # First try inventory match
        inventory_match = self.vector_search(ingredient)
        if inventory_match and inventory_match.get('similarity', 0) >= 0.5:
            return inventory_match
            
        # If no inventory match, try retail prices
        try:
            best_price, best_retailer = self.price_scraper.get_best_price(ingredient['item'])
            if best_price:
                # Convert scraper result to match format
                retail_match = {
                    'inventory_item': ingredient['item'],
                    'cost_per_unit': Decimal(str(best_price['total_price'] / best_price['total_quantity'])),
                    'unit': best_price['unit'],
                    'supplier': f"Retail ({best_retailer.capitalize()})",
                    'is_retail_estimate': True
                }
                return retail_match
        except Exception as e:
            return None
            
        return None

class UnitConverter:
    """Smart unit conversion with GPT fallback."""
    
    def __init__(self, openai_client=None):
        self.openai_client = openai_client
        self.ureg = pint.UnitRegistry()
    
    def convert(self, amount: Decimal, from_unit: str, to_unit: str) -> Decimal:
        """Convert between units using Pint with GPT fallback."""
        try:
            # Try Pint conversion first
            quantity = float(amount) * self.ureg(self.standardize_unit(from_unit))
            converted = quantity.to(self.standardize_unit(to_unit))
            return Decimal(str(converted.magnitude))
        except Exception as e:
            if self.openai_client:
                return self.gpt_conversion(amount, from_unit, to_unit)
            raise
    
    def gpt_conversion(self, amount: Decimal, from_unit: str, to_unit: str) -> Decimal:
        """Use GPT for unit conversion."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Convert between units. Return only the numerical conversion factor as a decimal."},
                    {"role": "user", "content": f"Convert 1 {from_unit} to {to_unit}"}
                ]
            )
            conversion_factor = Decimal(response.choices[0].message.content.strip())
            return amount * conversion_factor
        except Exception as e:
            raise
    
    @staticmethod
    def standardize_unit(unit: str) -> str:
        """Standardize unit notation."""
        unit_map = {
            'tbsp': 'tablespoon',
            'tsp': 'teaspoon',
            'cup': 'cup',
            'oz': 'ounce',
            'lb': 'pound',
            'g': 'gram',
            'kg': 'kilogram',
            'ml': 'milliliter',
            'l': 'liter',
            'gallon': 'gallon'
        }
        return unit_map.get(unit.lower().strip(), unit)

class RecipeCostCalculator:
    """Calculate recipe costs with smart matching and conversion."""
    
    def __init__(self, search_client, openai_client):
        self.matcher = IngredientMatcher(search_client, openai_client)
        self.converter = UnitConverter(openai_client)
    
    async def calculate_ingredient_cost(self, ingredient: Dict, match: Dict) -> Dict:
        """Calculate cost for single ingredient."""
        try:
            recipe_amount = Decimal(str(ingredient['amount']))
            recipe_unit = ingredient['unit']
            inventory_unit = match['unit']
            
            # Convert units
            converted_amount = self.converter.convert(
                recipe_amount,
                recipe_unit,
                inventory_unit
            )
            
            unit_cost = Decimal(str(match['cost_per_unit']))
            total_cost = converted_amount * unit_cost
            
            return {
                'ingredient': ingredient['item'],
                'recipe_amount': f"{recipe_amount} {recipe_unit}",
                'inventory_item': match['inventory_item'],
                'inventory_unit': inventory_unit,
                'converted_amount': float(converted_amount),
                'unit_cost': float(unit_cost),
                'total_cost': float(total_cost),
                'is_retail_estimate': match.get('is_retail_estimate', False)
            }
        except Exception as e:
            return None
    
    async def calculate_recipe_cost(self, recipe: Dict) -> Dict:
        """Calculate total recipe cost."""
        if not isinstance(recipe, dict) or 'name' not in recipe or 'ingredients' not in recipe:
            raise ValueError("Invalid recipe format")
            
        ingredient_costs = []
        total_cost = Decimal('0')
        
        for ingredient in recipe['ingredients']:
            # Get best match
            match = await self.matcher.get_best_match(ingredient)
            if not match:
                continue
            
            # Calculate cost
            cost_info = await self.calculate_ingredient_cost(ingredient, match)
            if cost_info:
                ingredient_costs.append(cost_info)
                total_cost += Decimal(str(cost_info['total_cost']))
        
        return {
            'recipe_name': recipe['name'],
            'ingredients': ingredient_costs,
            'total_cost': float(total_cost),
            'topping': recipe.get('topping', '')
        }
    
def extract_recipes_from_pdf(pdf_path, form_client, openai_client)-> dict:
    """Extract recipes from PDF."""
    with open(pdf_path, "rb") as doc:
        result = form_client.begin_analyze_document("prebuilt-layout", doc).result()
        content = {
            "tables": [[cell.content.strip() for cell in table.cells] for table in result.tables],
            "text": [p.content.strip() for p in result.paragraphs]
        }
    Systemprompt= """You are a precise recipe extraction specialist. Your task is to extract and standardize recipe information from any source while maintaining a consistent structure.

EXTRACTION RULES:
Extract all recipes from the content provided.
1. Extract ALL recipes from the provided content
2. Maintain exact measurements and units
3. Convert all text numbers to numeric values (e.g., "one" → 1)
4. Standardize ingredients to their base names
5. Capture complete procedures 

OUTPUT STRUCTURE:
Return data in this EXACT JSON format:
{
    "recipes": [
        {
            "name": "Complete Recipe Name with Size/Yield",
            "ingredients": [
                {
                    "item": "ingredient base name",
                    "amount": number,
                    "unit": "standardized unit"
                }
            ],
            "topping": "complete topping instructions"
            "size": "recipe size or yield"
            "preparation": "recipe preparation notes"
        }
    ]
}

STANDARDIZATION RULES:

1. Units: Use these standard units ONLY:Make sure to Convert to abbreviations
   Volume:
   - "ml" (milliliters)
   - "l" (liters)
   - "oz" (fluid ounces)
   - "cup" (cups)
   - "tbsp" (tablespoons)
   - "tsp" (teaspoons)
   - "gallon" (gallons)
   - " cc" (cubic centimeters)
   
   Weight:
   - "g" (grams)
   - "kg" (kilograms)
   - "lb" (pounds)
   - "oz" (ounces for weight)

2. Numbers:
   - Convert all written numbers to numerals
   - Convert fractions to decimals
   - Round to 2 decimal places
   - Examples:
     * "one" → 1
     * "half" → 0.5
     * "1/4" → 0.25
     * "2 1/2" → 2.5

3. Ingredients:
   - Use base ingredient names
   - Include preparation state in name if critical
   - Examples:
     * "pure vanilla extract" → "vanilla extract"
     * "cold butter, cubed" → "butter"

4. Measurements:
   - Convert all measurements to standard units
   - Handle common conversions:
     * "stick of butter" → 0.5, "cup"
     * "large egg" → 1, "unit"
     * "pinch" → 0.125, "tsp"

5. Topping Instructions:
   - Include complete application method
   - Maintain sequence of steps
   - Include any critical timing or temperature notes

VALIDATION REQUIREMENTS:
1. Every ingredient MUST have:
   - Non-empty "item" name
   - Numeric "amount"
   - Valid "unit" from standardized list

2. Every recipe MUST have:
   - Complete "name"
   - At least one ingredient
   - Either topping instructions or empty string ""

3. Numbers:
   - All amounts must be positive numbers
   - No text-based numbers allowed
   - No ranges (use average if range given)

HANDLING SPECIAL CASES:
1. Missing Measurements:
   - For "to taste" → use minimum recommended amount
   - For "as needed" → use typical serving amount
   - For decorative items → use minimum functional amount

2. Alternative Ingredients:
   - List primary ingredient only
   - Ignore "or" alternatives

3. Optional Ingredients:
   - Include in main list
   - Use minimum recommended amount

Return ONLY the JSON with no additional text or explanations."""

    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": Systemprompt},
            {"role": "user", "content": str(content)}
        ],
        response_format={"type": "json_object"}
    )
    logging.info(response.choices[0].message.content)
    
    try:
        recipes_data = json.loads(response.choices[0].message.content)
        return recipes_data
    except json.JSONDecodeError as e:
        raise
    except ValueError as e:
        raise

def export_to_excel(recipes_with_costs: List[Dict], output_path: Optional[str] = None) -> str:
    """Export recipe costs to Excel with detailed sheets."""
    try:
        if output_path is None:
            # Generate default output path in current directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"recipe_costs_{timestamp}.xlsx"
            
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
        os.makedirs(output_dir, exist_ok=True)
        
        with pd.ExcelWriter(output_path) as writer:
            # Recipe Summary Sheet
            summary_data = []
            for recipe in recipes_with_costs:
                summary_data.append({
                    'Recipe Name': recipe['recipe_name'],
                    'Total Cost': f"${recipe['total_cost']:.2f}",
                    'Ingredient Count': len(recipe['ingredients']),
                    'Topping': recipe['topping'],
                    'Size/Yield': recipe.get('size', ''),
                    'preparation': recipe.get('preparation', '')
                })
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Recipe Summary', index=False)
            
            # Detailed Ingredients Sheet
            ingredient_data = []
            for recipe in recipes_with_costs:
                for ing in recipe['ingredients']:
                    ingredient_data.append({
                        'Recipe Name': recipe['recipe_name'],
                        'Ingredient': ing['ingredient'],
                        'Recipe Amount': ing['recipe_amount'],
                        'Inventory Item': ing['inventory_item'],
                        'Converted Amount': f"{ing['converted_amount']:.3f} {ing['inventory_unit']}",
                        'Unit Cost': f"${ing['unit_cost']:.3f}",
                        'Total Cost': f"${ing['total_cost']:.2f}"
                    })
            pd.DataFrame(ingredient_data).to_excel(writer, sheet_name='Ingredient Details', index=False)
        
        return output_path
        
    except Exception as e:
        raise

def extract_recipes_from_excel(excel_path: str, openai_client) -> dict:
    """Extract recipes from Excel file."""
    try:
        # Read Excel file
        df = pd.read_excel(excel_path)
        
        # Convert DataFrame to text format similar to Form Recognizer output
        content = {
            "tables": [df.values.tolist()],  # Convert DataFrame to list of lists
            "text": df.to_string().split('\n')  # Convert to text format
        }
        
        # Use the same GPT prompt as before to maintain consistency
        system_prompt = """You are a precise recipe extraction specialist. Your task is to extract and standardize recipe information from any source while maintaining a consistent structure.

EXTRACTION RULES:
Extract all recipes from the content provided.
1. Extract ALL recipes from the provided content
2. Maintain exact measurements and units
3. Convert all text numbers to numeric values (e.g., "one" → 1)
4. Standardize ingredients to their base names
5. Capture complete procedures 

OUTPUT STRUCTURE:
Return data in this EXACT JSON format:
{
    "recipes": [
        {
            "name": "Complete Recipe Name with Size/Yield",
            "ingredients": [
                {
                    "item": "ingredient base name",
                    "amount": number,
                    "unit": "standardized unit"
                }
            ],
            "topping": "complete topping instructions"
            "size": "recipe size or yield"
            "preparation": "recipe preparation notes"
        }
    ]
}

STANDARDIZATION RULES:
1. Units: Use these standard units ONLY:
   Volume:
   - "ml" (milliliters)
   - "l" (liters)
   - "oz" (fluid ounces)
   - "cup" (cups)
   - "tbsp" (tablespoons)
   - "tsp" (teaspoons)
   - "gallon" (gallons)
   - "cc" (cubic centimeters)
   
   Weight:
   - "g" (grams)
   - "kg" (kilograms)
   - "lb" (pounds)
   - "oz" (ounces for weight)

2. Numbers:
   - Convert all written numbers to numerals
   - Convert fractions to decimals
   - Round to 2 decimal places

3. Ingredients:
   - Use base ingredient names
   - Include preparation state in name if critical

4. Measurements:
   - Convert all measurements to standard units
   - Handle common conversions

5. Topping Instructions:
   - Include complete application method
   - Maintain sequence of steps"""
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(content)}
            ],
            response_format={"type": "json_object"}
        )
        
        try:
            recipes_data = json.loads(response.choices[0].message.content)
            return recipes_data
        except json.JSONDecodeError as e:
            raise
        except ValueError as e:
            raise
            
    except Exception as e:
        raise
async def process_recipe_folder(folder_path: str, calculator: RecipeCostCalculator, form_client: DocumentAnalysisClient, openai_client: OpenAI):
    all_recipes = []
    
    try:
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
            
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            try:
                recipes_data = None
                
                if is_excel_file(file_path):
                    recipes_data = extract_recipes_from_excel(file_path, openai_client)
                    print(f"Extracted recipes from Excel: {json.dumps(recipes_data, indent=2)}")
                    
                elif is_pdf_file(file_path) or is_image_file(file_path):
                    recipes_data = extract_recipes_from_pdf(file_path, form_client, openai_client)
                    print(f"Extracted recipes from PDF/Image: {json.dumps(recipes_data, indent=2)}")
                    
                else:
                    continue
                
                if recipes_data and recipes_data.get('recipes'):
                    for recipe in recipes_data['recipes']:
                        try:
                            print(f"Processing recipe: {recipe['name']}")
                            cost_info = await calculator.calculate_recipe_cost(recipe)
                            if cost_info:
                                print(f"Got cost info: {json.dumps(cost_info, indent=2)}")
                                all_recipes.append(cost_info)
                            else:
                                print(f"No cost info generated for recipe: {recipe['name']}")
                        except Exception as e:
                            print(f"Error calculating recipe cost: {str(e)}")
                            continue
            
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                continue
                
        return all_recipes
        
    except Exception as e:
        print(f"Error in process_recipe_folder: {str(e)}")
        return []
def is_image_file(file_path: str) -> bool:
    """Check if file is an image based on its mimetype."""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type is not None and mime_type.startswith('image/')

def is_pdf_file(file_path: str) -> bool:
    """Check if file is a PDF."""
    return file_path.lower().endswith('.pdf')

def is_excel_file(file_path: str) -> bool:
    """Check if file is an Excel file."""
    return file_path.lower().endswith(('.xlsx', '.xls'))    



async def main():
    """Run the recipe cost calculator on a folder of files."""
    try:
        # Setup clients
        search_client = SearchClient(
            endpoint=os.getenv("AZURE_AISEARCH_ENDPOINT"),
            index_name="drift-customer",
            credential=AzureKeyCredential(os.getenv("AZURE_AISEARCH_APIKEY"))
        )
        
        form_client = DocumentAnalysisClient(
            endpoint=os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("AZURE_FORM_RECOGNIZER_KEY"))
        )
        
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize calculator
        calculator = RecipeCostCalculator(search_client, openai_client)
        
        # Process recipe folder
        #"C:\Users\rahul\Downloads\Bulk Recipes and Sub-recipes-20241205T074727Z-001"
        folder_path = "C:/Users/rahul/Downloads/New added menu items 11-26-2024-20241205T074732Z-001/New added menu items 11-26-2024"  # Update this path
        recipe_costs = await process_recipe_folder(folder_path, calculator, form_client, openai_client)
        
        # Generate output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(folder_path, f"recipe_costs_{timestamp}.xlsx")
        
        # Export results
        final_path = export_to_excel(recipe_costs, output_path)
        print(f"Results saved to: {final_path}")
        
    except Exception as e:
        raise

if __name__ == "__main__":
    asyncio.run(main())