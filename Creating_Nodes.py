import json
from neo4j import GraphDatabase

# Connect to Neo4j database
uri = "bolt://localhost:7687"  # Change this based on your setup
username = "neo4j"
password = "1234567890"  # Replace with your password
driver = GraphDatabase.driver(uri, auth=(username, password))

# Function to create a Product node in Neo4j
def create_product(tx, product):

    total_comments = product.get("total_comments", "")
    
    # Check if total_comments is not an integer and apply replace if needed
    if isinstance(total_comments, str):
        total_comments = total_comments.replace("(", "").replace(")", "")
    
    # Create Product node
    product_query = """
    CREATE (p:Product {
        name: $name, 
        origin_price : $origin_price,
        discount : $discount, 
        price: $price,
        rating: $rating, 
        detail: $detail,
        detail_2: $detail_2,
        detail_3: $detail_3,
        total_comments: $total_comments,
        image: $image,
        detail_link: $detail_link
    })
    RETURN id(p) AS product_id
    """
    result = tx.run(product_query, 
                    name=product.get("Product", ""), 
                    origin_price = product.get("Origin_price" , ""),
                    discount = product.get("Discount" , ""),
                    price=product.get("Price", ""),
                    rating=product.get("rating", ""),
                    detail=product.get("detail", ""),
                    detail_2=product.get("detail_2", ""),
                    detail_3=product.get("detail_3", ""),
                    total_comments=total_comments,
                    image=product.get("Image", ""),
                    detail_link=product.get("detail_link", "")
                   )
    
    # Get the created product node's ID
    product_id = result.single()["product_id"]
    
    # Create Comment nodes and link them to the product using the product ID
    top_comments = product.get("top_comments", [])
    for comment in top_comments:
        comment_query = """
        MATCH (p:Product) WHERE id(p) = $product_id
        CREATE (c:Comment {
            head_comment: $head_comment, 
            detail_comment: $detail_comment
        })
        CREATE (p)-[:HAS_COMMENT]->(c)
        """
        tx.run(comment_query, 
               head_comment=comment["head_comment"], 
               detail_comment=comment.get("detail_comment", ""),
               product_id=product_id
              )

# Load JSON data

for i in range(1,25):
        with open(f'results{i}.json', encoding='utf-8') as f:  # Update path to your JSON file
            data = json.load(f)

        # Create nodes in Neo4j
        with driver.session() as session:
            for product in data:
                session.execute_write(create_product, product)


# Close the driver connection
driver.close()
