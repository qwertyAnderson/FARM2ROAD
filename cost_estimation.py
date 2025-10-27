def calculate_cost(distance_km, base_rate=10, num_farmers=1):
    
    solo_cost = distance_km * base_rate
    pooled_cost = solo_cost / num_farmers
    return solo_cost, pooled_cost


distance = float(input("Enter distance (in km): "))
farmers = int(input("Enter number of farmers sharing vehicle: "))

solo, pooled = calculate_cost(distance, 10, farmers)

print("\n=== COST ESTIMATION REPORT ===")
print(f"Distance: {distance} km")
print(f"Base Rate: ₹10 per km")
print(f"Solo Delivery Cost: ₹{solo}")
print(f"Pooled Delivery Cost (per farmer): ₹{pooled}")
print(f"Total Savings per farmer: ₹{solo - pooled}")
