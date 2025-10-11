using Plots

# A call option

Call(S, K) = max(S - K, 0)

# A put option

Put(S, K) = max(K - S, 0)


# Asuming a K = 100

K = 100


plot(S -> Call(S, K), 0, 200, label="Call", title="Call and Put Options", xlabel="Stock Price", ylabel="Option Price")
plot!(S -> Put(S, K), 0, 200, label="Put")


# 3D Surface Plot - Option Value vs Stock Price and Strike Price
S_range = 50:2:150  # Stock prices
K_range = 80:2:120  # Strike prices

# Create a surface plot for call option values
surface(S_range, K_range, (S, K) -> Call(S, K), 
        xlabel="Stock Price (S)", 
        ylabel="Strike Price (K)", 
        zlabel="Call Option Value",
        title="Call Option Value Surface",
        camera=(25, 20))