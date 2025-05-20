module fc_784to256 (
    input wire clk,
    input wire rst_n,
    input wire i_valid,
    input wire [783:0] i_data,
    input wire [783:0] i_weight,
    input wire [9:0] i_threshold,
    output reg o_result,
    output reg o_valid
);

    // Stage 0
    reg [783:0] r_data_s0, r_weight_s0;
    reg [9:0]   r_threshold_s0;
    reg         valid_s0;

    // Stage 1
    reg [783:0] r_xnor_s1;
    reg         valid_s1;

    // Stage 2
    reg [9:0]   r_popcount_s2;
    reg [9:0]   r_threshold_s2;
    reg         valid_s2;

    // ------------------------
    // Stage 0: Capture input
    // ------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_data_s0      <= 0;
            r_weight_s0    <= 0;
            r_threshold_s0 <= 0;
            valid_s0       <= 0;
        end else begin
            if (i_valid) begin
                r_data_s0      <= i_data;
                r_weight_s0    <= i_weight;
                r_threshold_s0 <= i_threshold;
                valid_s0       <= 1;
            end else begin
                valid_s0 <= 0;
            end
        end
    end

    // ------------------------
    // Stage 1: XNOR
    // ------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_xnor_s1 <= 0;
            valid_s1  <= 0;
        end else begin
            if (valid_s0) begin
                r_xnor_s1 <= ~(r_data_s0 ^ r_weight_s0);
                valid_s1  <= 1;
            end else begin
                valid_s1 <= 0;
            end
        end
    end

    // ------------------------
    // Stage 2: Popcount + Threshold
    // ------------------------
    function automatic [9:0] count_ones(input [783:0] vec);
        integer i;
        begin
            count_ones = 0;
            for (i = 0; i < 784; i = i + 1)
                count_ones = count_ones + vec[i];
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_popcount_s2   <= 0;
            r_threshold_s2 <= 0;
            valid_s2     <= 0;
        end else begin
            if (valid_s1) begin
                r_popcount_s2     <= count_ones(r_xnor_s1);
                r_threshold_s2    <= r_threshold_s0;  // from stage 0
                valid_s2          <= 1;
            end else begin
                valid_s2 <= 0;
            end
        end
    end

    // ------------------------
    // Stage 3: Final compare
    // ------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            o_result <= 0;
            o_valid  <= 0;
        end else begin
            if (valid_s2) begin
                o_result <= (r_popcount_s2 >= r_threshold_s2);
                o_valid  <= 1;
            end else begin
                o_valid <= 0;
            end
        end
    end
endmodule
