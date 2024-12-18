import json
import pandas as pd
from typing import List, Dict, Optional, Tuple
from scrapegraphai.graphs import SmartScraperGraph
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class InventoryItem:
    inventory_item_name: str
    measured_in: str = ""
    cost_of_unit: Optional[Decimal] = None
    total_price: Optional[Decimal] = None
    total_quantity: Optional[Decimal] = None
    source_retailer: str = ""  # Added to track which retailer provided the best price

    def calculate_cost_per_unit(self) -> None:
        """Calculate cost per unit from total price and quantity"""
        try:
            if self.total_price and self.total_quantity and self.total_quantity != 0:
                self.cost_of_unit = (self.total_price / self.total_quantity).quantize(
                    Decimal('0.01'), rounding=ROUND_HALF_UP
                )
                print(f"Calculation from {self.source_retailer}: ${self.total_price} / {self.total_quantity} {self.measured_in} = ${self.cost_of_unit} per {self.measured_in}")
        except (InvalidOperation, TypeError, ZeroDivisionError) as e:
            print(f"Error calculating cost per unit: {e}")
            self.cost_of_unit = None

class MultiRetailerPriceScraper:
    def __init__(self):
        self.config = {
            "llm": {
                "api_key": "API_KEY",
                "model": "openai/gpt-4o",
            },
            "verbose": True,
            "headless": False
        }
        self.retailers = {
            "google": "https://www.google.com/search?q={query}+price",
            "amazon": "https://www.amazon.com/s?k={query}",
            "walmart": "https://www.walmart.com/search?q={query}",
            "costco": "https://www.costco.com/CatalogSearch?keyword={query}"
        }

    def _create_prompt(self, inventory_name: str, retailer: str) -> str:
        return f"""
        For this product from {retailer.capitalize()}: {inventory_name}
        Find and extract:
        1. The total price of the package
        2. The total quantity in the package
        3. The unit of measure
        
        Return ONLY this JSON format with numerical values (no currency symbols or units in values):
        {{
            "total_price": 13.00,
            "total_quantity": 30,
            "unit": "pounds",
            "retailer": "{retailer}"
        }}
        
        Example 1: For a chicken package that costs $13.00 for 30 lbs:
        {{
            "total_price": 13.00,
            "total_quantity": 30,
            "unit": "pounds",
            "retailer": "{retailer}"
        }}
        
        Example 2: For spices that cost $12.00 for 24 oz:
        {{
            "total_price": 12.00,
            "total_quantity": 24,
            "unit": "ounce",
            "retailer": "{retailer}"
        }}
        
        Only return exact matches with clear prices and quantities.
        Skip items with unclear or estimated values.
        """

    def safe_decimal_convert(self, value: any) -> Optional[Decimal]:
        """Safely convert a value to Decimal"""
        try:
            if value is None:
                return None
            if isinstance(value, str):
                value = value.replace('$', '').strip()
            return Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            return None

    def scrape_retailer(self, inventory_name: str, retailer: str) -> Optional[Dict]:
        """Scrape price data from a specific retailer"""
        url = self.retailers[retailer].format(query=inventory_name.replace(' ', '+'))
        try:
            graph = SmartScraperGraph(
                prompt=self._create_prompt(inventory_name, retailer),
                source=url,
                config=self.config
            )
            result = graph.run()
            
            if isinstance(result, dict) and result.get('total_price') != 'NA':
                result['retailer'] = retailer
                print(f"Found price data from {retailer}:", json.dumps(result, indent=2))
                return result
            
        except Exception as e:
            print(f"Error scraping {retailer}: {e}")
        
        return None

    def get_best_price(self, inventory_name: str) -> Tuple[Optional[Dict], str]:
        """Get the best price across all retailers"""
        best_price = None
        best_retailer = None
        min_cost_per_unit = float('inf')

        # Use ThreadPoolExecutor to scrape prices concurrently
        with ThreadPoolExecutor(max_workers=len(self.retailers)) as executor:
            future_to_retailer = {
                executor.submit(self.scrape_retailer, inventory_name, retailer): retailer
                for retailer in self.retailers
            }

            for future in as_completed(future_to_retailer):
                retailer = future_to_retailer[future]
                try:
                    result = future.result()
                    if result:
                        price = self.safe_decimal_convert(result['total_price'])
                        quantity = self.safe_decimal_convert(result['total_quantity'])
                        
                        if price and quantity and quantity != 0:
                            cost_per_unit = float(price / quantity)
                            if cost_per_unit < min_cost_per_unit:
                                min_cost_per_unit = cost_per_unit
                                best_price = result
                                best_retailer = retailer
                
                except Exception as e:
                    print(f"Error processing {retailer} result: {e}")

        return best_price, best_retailer

    def update_row_with_prices(self, row: pd.Series, scraped_data: Dict) -> pd.Series:
        """Update a single row with scraped price data"""
        try:
            total_price = self.safe_decimal_convert(scraped_data.get('total_price'))
            total_quantity = self.safe_decimal_convert(scraped_data.get('total_quantity'))
            unit = str(scraped_data.get('unit', '')).lower().strip()
            retailer = scraped_data.get('retailer', '')

            if total_price and total_quantity and unit:
                item = InventoryItem(
                    inventory_item_name=row['Inventory Item Name'],
                    measured_in=unit,
                    total_price=total_price,
                    total_quantity=total_quantity,
                    source_retailer=retailer
                )
                item.calculate_cost_per_unit()

                if item.cost_of_unit:
                    row['Measured In'] = unit
                    row['Cost of a Unit'] = float(item.cost_of_unit)
                    row['Last Updated At'] = datetime.datetime.now().isoformat()
                    row['Source Retailer'] = retailer.capitalize()
                    print(f"Updated {row['Inventory Item Name']}: ${item.cost_of_unit} per {unit} from {retailer}")

        except Exception as e:
            print(f"Error updating row: {e}")

        return row

    def update_inventory_prices(self, csv_file: str) -> pd.DataFrame:
        """Update inventory prices while preserving all original columns"""
        print("Reading inventory file...")
        df = pd.read_csv(csv_file)
        if 'Source Retailer' not in df.columns:
            df['Source Retailer'] = ''
        print(f"Found {len(df)} items to process")

        for idx, row in df.iterrows():
            inventory_name = str(row['Inventory Item Name']).strip()
            if not inventory_name:
                continue

            print(f"\nProcessing ({idx + 1}/{len(df)}): {inventory_name}")
            best_price, best_retailer = self.get_best_price(inventory_name)
            
            if best_price:
                df.iloc[idx] = self.update_row_with_prices(row, best_price)
            else:
                print(f"No valid price data found for {inventory_name} across any retailer")

        return df
    async def find_ingredient_retail_price(ingredient: Dict, api_key: str) -> Optional[Dict]:
    # For testing purposes, return a properly formatted dictionary
        return {
        'inventory_item': ingredient['item'],
        'cost_per_unit': Decimal('1.99'),
        'unit': ingredient['unit'],
        'supplier': 'Retail Estimate',
        'is_retail_estimate': True
     }

def main():
    # File paths
    input_csv = r"C:\Users\rahul\OneDrive\Desktop\processed_inventory_sam's.csv"
    output_excel = "updated_inventory_prices_multi_retailer.xlsx"
    output_json = "updated_inventory_prices_multi_retailer.json"
    
    try:
        # Create scraper and update prices
        scraper = MultiRetailerPriceScraper()
        updated_df = scraper.update_inventory_prices(input_csv)
        
        # Save to JSON
        print("\nSaving results to JSON...")
        results = updated_df.to_dict(orient='records')
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        
        # Save to Excel with formatting
        print("Saving results to Excel...")
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            updated_df.to_excel(writer, index=False, sheet_name='Updated Inventory')
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Updated Inventory']
            for idx, col in enumerate(updated_df.columns):
                max_length = max(
                    updated_df[col].astype(str).apply(len).max(),
                    len(col)
                )
                worksheet.column_dimensions[chr(65 + idx)].width = max_length + 2
        
        
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()